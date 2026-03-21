"""
ZipperGen — real-time web viewer (ZipperChat).

Serves a single-page visualisation at http://localhost:PORT showing the
agent conversation as a multi-column diagram, updated live via
Server-Sent Events (SSE). No external dependencies — only stdlib.

Usage
-----
    from zipperchat import WebTrace
    from zippergen.runtime import run

    wt = WebTrace(program.lifelines).start()

    while True:
        wt.reset()
        run(wf, list(program.lifelines), initial, trace=wt)
        wt.done()
        wt.wait_for_replay()   # blocks until the ▶ button is clicked
"""

from __future__ import annotations

import json
import pathlib
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_ASSETS = pathlib.Path(__file__).parent / "assets"

__all__ = ["WebTrace"]


# ---------------------------------------------------------------------------
# Event bus — fan-out to all connected SSE clients, with replay for late joins
# ---------------------------------------------------------------------------

class _EventBus:
    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._history: list[dict] = []

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            for event in self._history:
                q.put(event)
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def publish(self, event: dict) -> None:
        with self._lock:
            if event.get("type") != "close":
                self._history.append(event)
            for q in self._subs:
                q.put(event)

    def reset(self) -> None:
        """Clear history; new events will flow through existing connections."""
        with self._lock:
            self._history.clear()


# ---------------------------------------------------------------------------
# HTTP handler — serves the viewer HTML and the /events SSE stream
# ---------------------------------------------------------------------------

def _make_handler(bus: _EventBus, lifelines: list[str],
                  replay_event: threading.Event,
                  init_event: dict):
    class _Handler(BaseHTTPRequestHandler):

        def do_GET(self):
            if self.path == "/":
                body = _HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                q = bus.subscribe()
                try:
                    while True:
                        event = q.get()
                        self._sse(event)
                        if event.get("type") == "close":
                            break
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    bus.unsubscribe(q)

            elif self.path.startswith("/assets/"):
                fname = self.path[len("/assets/"):]
                fpath = _ASSETS / fname
                if fpath.exists() and fpath.is_file():
                    body = fpath.read_bytes()
                    ctype = "image/svg+xml" if fname.endswith(".svg") else "application/octet-stream"
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/replay":
                # Reset bus history and publish fresh init *before* responding,
                # so a reconnecting browser gets a clean slate from history.
                bus.reset()
                bus.publish(init_event)
                replay_event.set()
                self.send_response(204)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def _sse(self, event: dict) -> None:
            msg = f"data: {json.dumps(event)}\n\n".encode()
            self.wfile.write(msg)
            self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:
            pass   # suppress request logging

    return _Handler


class _Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass   # suppress connection-reset noise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class WebTrace:
    """
    Callable trace that feeds events to the browser viewer via SSE.

    Pass an instance as ``trace=`` to ``run()``.

    Typical loop::

        wt = WebTrace(program.lifelines).start()
        while True:
            wt.reset()
            run(wf, lifelines, initial, trace=wt)
            wt.done()
            wt.wait_for_replay()
    """

    def __init__(self, lifelines, port: int = 8765):
        self._lifelines = [l.name if hasattr(l, "name") else str(l) for l in lifelines]
        self._port = port
        self._bus = _EventBus()
        self._server: ThreadingHTTPServer | None = None
        self._replay_event = threading.Event()

    def start(self) -> "WebTrace":
        init_ev = {"type": "init", "lifelines": self._lifelines}
        handler = _make_handler(self._bus, self._lifelines, self._replay_event, init_ev)
        self._server = _Server(("", self._port), handler)
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        print(f"ZipperChat → http://localhost:{self._port}")
        self._bus.publish(init_ev)
        return self

    def reset(self) -> "WebTrace":
        """Clear the diagram and prepare for a new run."""
        self._replay_event.clear()
        self._bus.reset()
        self._bus.publish({"type": "init", "lifelines": self._lifelines})
        return self

    def wait_for_replay(self) -> None:
        """Block until the ▶ Run again button is clicked in the browser."""
        self._replay_event.wait()
        self._replay_event.clear()

    def __call__(self, event: dict) -> None:
        self._bus.publish(event)

    def done(self) -> None:
        self._bus.publish({"type": "done"})

    def stop(self) -> None:
        self._bus.publish({"type": "close"})
        if self._server:
            self._server.shutdown()


# ---------------------------------------------------------------------------
# Embedded viewer HTML
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --ui-zoom: 1.1; }
  html  { zoom: var(--ui-zoom); }
</style>
<title>ZipperChat</title>
<style>
/* ── Reset & base ─────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter',
               Helvetica, Arial, sans-serif;
  background: #f6f3eb;
  color: #24313d;
  display: flex;
  flex-direction: column;
  height: calc(100vh / var(--ui-zoom));  /* compensate so body fills exactly the viewport after zoom */
  overflow: hidden;
}

/* ── Top bar ─────────────────────────────────────────────────────────── */
#topbar {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 0 20px 0 30px;
  height: 56px;
  background: #fbf8f2;
  border-bottom: 1px solid #dccfb9;
  flex-shrink: 0;
}

.logo-wrap {
  display: flex;
  align-items: center;
  height: 44px;
  overflow: hidden;
  flex-shrink: 0;
  margin-left: 5px;
}
.logo-wrap svg { display: block; }

.brand {
  display: flex;
  flex-direction: column;
  line-height: 1.2;
}
.brand-main {
  font-size: 17px;
  font-weight: 700;
  color: #24313d;
  letter-spacing: -0.3px;
}
.brand-sub {
  font-size: 10px;
  font-weight: 500;
  color: #7b6f61;
  letter-spacing: 0.5px;
}

#status {
  margin-left: auto;
  margin-right: 18px;
  padding: 4px 12px;
  border-radius: 20px;
  background: #efe7d8;
  color: #7b6f61;
  font-size: 12px;
  font-weight: 500;
}
#status.running { background: #e3f3e8; color: #177245; }
#status.done    { background: #e4eefb; color: #215ea6; }

/* ── Replay button ───────────────────────────────────────────────────── */
#replay-btn {
  padding: 6px 16px;
  margin-right: 18px;
  border-radius: 20px;
  border: 1px solid #C8860A;
  background: transparent;
  color: #C8860A;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  letter-spacing: 0.3px;
  transition: background 0.15s, color 0.15s;
}
#replay-btn:hover {
  background: #C8860A;
  color: #fbf8f2;
}

/* ── Scroll container ─────────────────────────────────────────────────── */
#container {
  flex: 1;
  overflow-y: auto;
  padding: 0 28px 60px;
}

/* ── Header row — sticky inside the scroll container ─────────────────── */
/* Placing the header inside #container means the header grid and every    */
/* row-wrapper grid share the same containing block, so 1fr resolves to    */
/* exactly the same pixel width and columns stay aligned at all times.     */
#diagram-header {
  position: sticky;
  top: 0;
  z-index: 10;          /* above event rows, which create stacking contexts via animation */
  display: grid;
  /* grid-template-columns set dynamically */
  column-gap: 0;
  padding-top: 24px;
  padding-bottom: 4px;  /* covers the gap between header and first event row */
  background: #f6f3eb;
}
#container::-webkit-scrollbar { width: 6px; }
#container::-webkit-scrollbar-track { background: transparent; }
#container::-webkit-scrollbar-thumb { background: #c9bda9; border-radius: 3px; }

/* ── Diagram column (events only) ────────────────────────────────────── */
#diagram {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

/* ── Column header cells ─────────────────────────────────────────────── */
.ll-header {
  padding: 9px 16px;
  border-radius: 10px 10px 0 0;
  border: 1px solid #d8cfbf;
  border-bottom: 2px solid;  /* accent border, set per column */
  text-align: center;
  font-weight: 700;
  font-size: 13px;
  margin: 0 3px 10px;
  letter-spacing: 0.2px;
  background: rgba(255, 252, 246, 0.9);
  box-shadow: 0 1px 0 rgba(120, 97, 64, 0.04);
}

/* ── Act row cells ────────────────────────────────────────────────────── */
.act-cell {
  padding: 3px 6px;
  min-height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  margin: 0 3px;
}

/* ── Act box ─────────────────────────────────────────────────────────── */
.act-box {
  width: 100%;
  max-width: 230px;
  background: #fffdfa;
  border: 1px solid #d8cfbf;
  border-left: 3px solid;   /* accent, set per column */
  border-radius: 7px;
  padding: 8px 11px;
  animation: fadein 0.18s ease;
  box-shadow: 0 6px 16px rgba(84, 63, 37, 0.08);
}
.act-name {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 4px;
  color: #24313d;
}
.act-in {
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  font-size: 10px;
  color: #62584d;
  line-height: 1.5;
  word-break: break-all;
  white-space: pre-line;
}
.act-out {
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  font-size: 10px;
  color: #177245;
  line-height: 1.5;
  word-break: break-all;
  white-space: pre-line;
}
.expandable {
  margin-top: 2px;
}
.expandable-text {
  position: relative;
}
.expandable-text.collapsed {
  max-height: 6.3em;
  overflow: hidden;
}
.expandable-text.collapsed::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 1.8em;
  background: linear-gradient(to bottom, rgba(255,253,250,0), #fffdfa 88%);
  pointer-events: none;
}
.trace-toggle {
  margin-top: 2px;
  margin-bottom: 5px;
  padding: 0;
  border: 0;
  background: none;
  color: #215ea6;
  font-size: 10px;
  font-weight: 600;
  cursor: pointer;
}
.trace-toggle:hover {
  color: #17497f;
}

/* ── Act row wrapper ──────────────────────────────────────────────────── */
.act-row {
  display: grid;
  column-gap: 0;
  /* grid-template-columns set dynamically */
}

/* ── Message row wrapper ──────────────────────────────────────────────── */
.msg-row {
  display: grid;
  column-gap: 0;
  /* grid-template-columns set dynamically */
  position: relative;
  min-height: 56px;
  margin: 0;
  border-radius: 4px;
  overflow: visible;
  animation: fadein 0.15s ease;
}

/* ── Message cell ─────────────────────────────────────────────────────── */
.msg-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 6px 8px;
  min-height: 56px;
  border-radius: 4px;
  margin: 0 3px;
}

/* ── Message box (send or recv) ──────────────────────────────────────── */
.msg-box {
  background: #fffdfa;
  border: 1px solid #d8cfbf;
  border-left: 3px solid;  /* accent color set inline */
  border-radius: 7px;
  padding: 7px 11px;
  font-size: 11px;
  max-width: 180px;
  min-width: 60px;
  box-shadow: 0 6px 16px rgba(84, 63, 37, 0.07);
}
.msg-box .msg-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  margin-bottom: 3px;
  opacity: 0.7;
}
.msg-box .msg-vals {
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  font-size: 10px;
  color: #3b4753;
  word-break: break-all;
  line-height: 1.4;
  white-space: pre-line;
}

/* ── SVG arrow overlay ───────────────────────────────────────────────── */
.msg-row svg {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
  overflow: visible;
}

/* ── Animations ──────────────────────────────────────────────────────── */
@keyframes fadein {
  from { opacity: 0; transform: translateY(5px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes act-pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.45; }
}
.act-box.running {
  animation: fadein 0.18s ease, act-pulse 1.4s ease-in-out 0.18s infinite;
}
</style>
</head>
<body>

<!-- ── Top bar ────────────────────────────────────────────────────────── -->
<div id="topbar">
  <div class="logo-wrap">
    <!-- Robot head only, viewBox crops to x=26-168, y=10-186 -->
    <svg width="36" height="44" viewBox="26 10 142 176"
         fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M160.662 35.2H32.7875V41.6H160.662V35.2Z" fill="#1A2340"/>
      <path d="M160.662 35.2H32.7875V38.4H160.662V35.2Z" fill="#222D50"/>
      <path d="M141.481 16H51.9688V35.2H141.481V16Z" fill="#1A2340"/>
      <path d="M138.087 16H61.3625V19.2H138.087V16Z" fill="#2A3860"/>
      <path d="M141.481 32H51.9688V35.2H141.481V32Z" fill="#C8860A"/>
      <path d="M141.481 41.6H51.9688V99.2H141.481V41.6Z" fill="#F0D9B0"/>
      <path d="M58.3625 41.6H51.9688V99.2H58.3625V41.6Z" fill="#D9BC8F"/>
      <path d="M141.481 41.6H135.087V99.2H141.481V41.6Z" fill="#D9BC8F"/>
      <path d="M77.5437 54.4H64.7562V64H77.5437V54.4Z" fill="#1A1A2E"/>
      <path d="M67.9531 54.4H64.7562V57.6H67.9531V54.4Z" fill="white"/>
      <path d="M128.694 54.4H115.906V64H128.694V54.4Z" fill="#1A1A2E"/>
      <path d="M119.103 54.4H115.906V57.6H119.103V54.4Z" fill="white"/>
      <path d="M77.5437 49.6H61.5594V52.8H77.5437V49.6Z" fill="#3A2A1A"/>
      <path d="M131.891 46.4H115.906V49.6H131.891V46.4Z" fill="#3A2A1A"/>
      <path d="M115.906 80H77.5437V83.2H115.906V80Z" fill="#8A6040"/>
      <path d="M115.906 76.8H109.512V80H115.906V76.8Z" fill="#8A6040"/>
      <path d="M51.9687 54.4H45.575V70.4H51.9687V54.4Z" fill="#D9BC8F"/>
      <path d="M147.875 54.4H141.481V70.4H147.875V54.4Z" fill="#D9BC8F"/>
      <path d="M109.512 99.2H83.9375V112H109.512V99.2Z" fill="#E8CCA0"/>
      <path d="M167 112H26V183H167V112Z" fill="#1A2340"/>
      <path d="M96.725 112H83.9375V118.4H96.725V112Z" fill="#F0D9B0"/>
      <path d="M109.512 112H96.725V118.4H109.512V112Z" fill="#F0D9B0"/>
      <path d="M106.316 118.4H87.1343V124.8H106.316V118.4Z" fill="#F5E8C8"/>
      <path d="M99.922 124.8H93.5281V156.8H99.922V124.8Z" fill="#C8860A"/>
      <path d="M103.119 118.4H90.3312V124.8H103.119V118.4Z" fill="#A06808"/>
      <path d="M106.316 150.4H87.1343V156.8H106.316V150.4Z" fill="#C8860A"/>
      <path d="M101.52 144H91.9297V150.4H101.52V144Z" fill="#A06808"/>
      <path d="M167.056 112H147.875V182.4H167.056V112Z" fill="#1A2340"/>
    </svg>
  </div>

  <div class="brand">
    <span class="brand-main">Zipper<span style="color:#C8860A">Chat</span></span>
    <span class="brand-sub">powered by ZipperGen</span>
  </div>

  <span id="status">connecting…</span>
</div>

<!-- ── Scrolling events (header is sticky inside so columns stay aligned) ── -->
<div id="container">
  <div id="diagram-header"></div>
  <div id="diagram"></div>
</div>

<script>
// ── Column palette ─────────────────────────────────────────────────────
const COLS = [
  { bg: 'rgba(71, 125, 209, 0.08)', accent: '#2c6cc9', text: '#2b4c7e' },
  { bg: 'rgba(52, 145, 92, 0.08)',  accent: '#2a8b57', text: '#205d3e' },
  { bg: 'rgba(132, 95, 204, 0.08)', accent: '#7851c6', text: '#55318d' },
  { bg: 'rgba(207, 138, 44, 0.08)', accent: '#c8860a', text: '#8b5b08' },
  { bg: 'rgba(194, 82, 112, 0.08)', accent: '#c25270', text: '#8c3148' },
];

// ── State ──────────────────────────────────────────────────────────────
let lifelines = [];
let N = 0;
const diagram   = document.getElementById('diagram');
const headerEl  = document.getElementById('diagram-header');
const container = document.getElementById('container');
const statusEl  = document.getElementById('status');
const topbar    = document.getElementById('topbar');

// ── Constraint graph ────────────────────────────────────────────────────
// Each row (act or message) is a node with a level (vertical position).
// AddEdge(u,v) means u must appear above v.  Raise() propagates level
// increases transitively.  syncOrder() re-sorts DOM elements by level.
const cgRows    = {};         // id → CgRow
let   cgIdSeq   = 0;
const cgLast    = {};         // lifeline → CgRow (last materialized event)
const cgPending = {};         // lifeline → Set of CgRow (open msg rows, recv pending)
const cgOpen    = {};         // "A->B" → CgRow[] FIFO

const completed       = [];   // message rows that have received (for arrow redraw)
const pendingActBoxes = {};   // "lifeline:seq" → act-box DOM element

// ── Helpers ────────────────────────────────────────────────────────────
const col = i => COLS[i % COLS.length];
function lifelineIdx(name) { return lifelines.indexOf(name); }
function scrollDown() { container.scrollTop = container.scrollHeight; }

function fmt(v, truncateStrings = true) {
  if (v === null || v === undefined) return 'null';
  if (v === 'κ_ctrl') return null;
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'string') {
    return truncateStrings && v.length > 32 ? v.slice(0, 30) + '…' : v;
  }
  if (Array.isArray(v)) {
    const parts = v.map(item => fmt(item, truncateStrings)).filter(x => x !== null);
    return parts.length ? parts.join('\n') : null;
  }
  return String(v);
}

function fmtDict(d, truncateStrings = true) {
  return Object.entries(d)
    .map(([k, v]) => {
      const s = fmt(v, truncateStrings);
      return s !== null ? `${k} = ${s}` : null;
    })
    .filter(Boolean).join('\n');
}

function fmtList(arr, truncateStrings = true) {
  return (arr || []).map(v => fmt(v, truncateStrings)).filter(x => x !== null).join('\n');
}

function isCtrlVals(vals) {
  return Array.isArray(vals) && vals.includes('κ_ctrl');
}

function makeExpandableText(className, text) {
  const wrapper = document.createElement('div');
  wrapper.className = 'expandable';

  const body = document.createElement('div');
  body.className = `${className} expandable-text collapsed`;
  body.textContent = text;
  wrapper.appendChild(body);

  const btn = document.createElement('button');
  btn.className = 'trace-toggle';
  btn.type = 'button';
  btn.textContent = 'show more';
  btn.hidden = true;
  btn.onclick = () => {
    const expanded = !body.classList.contains('collapsed');
    body.classList.toggle('collapsed', expanded);
    btn.textContent = expanded ? 'show more' : 'show less';
  };
  wrapper.appendChild(btn);

  requestAnimationFrame(() => {
    const needsToggle = body.scrollHeight > body.clientHeight + 1;
    btn.hidden = !needsToggle;
    if (!needsToggle) body.classList.remove('collapsed');
  });

  return wrapper;
}

// ── Diagram init ───────────────────────────────────────────────────────
function clearDiagram() {
  diagram.innerHTML = '';
  headerEl.innerHTML = '';
  lifelines = []; N = 0;
  Object.keys(cgRows).forEach(k => delete cgRows[k]);
  cgIdSeq = 0;
  Object.keys(cgLast).forEach(k => delete cgLast[k]);
  Object.keys(cgPending).forEach(k => delete cgPending[k]);
  Object.keys(cgOpen).forEach(k => delete cgOpen[k]);
  Object.keys(pendingActBoxes).forEach(k => delete pendingActBoxes[k]);
  completed.length = 0;
}

function initDiagram(lls) {
  lifelines = lls;
  N = lls.length;
  headerEl.style.gridTemplateColumns = `repeat(${N}, 1fr)`;
  // #diagram is now a flex column; grid template is set per-row wrapper

  lls.forEach((name, i) => {
    const c = col(i);
    const h = document.createElement('div');
    h.className = 'll-header';
    h.textContent = name;
    h.style.borderBottomColor = c.accent;
    h.style.color = c.text;
    headerEl.appendChild(h);
  });
}

// ── Constraint graph helpers ────────────────────────────────────────────
function newCgRow(kind) {
  const r = { id: cgIdSeq++, kind, level: 0, succ: new Set(), pred: new Set(),
              wrapper: null, cells: null, svg: null,
              fromIdx: -1, toIdx: -1, ctrl: false, color: null,
              occupied: new Set() };
  cgRows[r.id] = r;
  return r;
}

function cgAddEdge(u, v) {
  if (!u || !v || u.id === v.id) return;
  if (u.succ.has(v.id)) return;
  u.succ.add(v.id); v.pred.add(u.id);
  if (v.level <= u.level) cgRaise(v, u.level + 1);
}

function cgRaise(v, nl) {
  if (nl <= v.level) return;
  v.level = nl;
  for (const wId of v.succ) cgRaise(cgRows[wId], v.level + 1);
}

function hasPath(start, target) {
  if (!start || !target) return false;
  if (start.id === target.id) return true;
  const seen = new Set([start.id]);
  const stack = [...start.succ];
  while (stack.length) {
    const id = stack.pop();
    if (id === target.id) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    const row = cgRows[id];
    if (!row) continue;
    for (const succ of row.succ) stack.push(succ);
  }
  return false;
}

function canReuseActRow(lifeline, r) {
  if (!r || r.kind !== 'action') return false;
  const idx = lifelineIdx(lifeline);
  if (r.occupied.has(idx)) return false;

  const last = cgLast[lifeline];
  if (last && hasPath(r, last)) return false;

  for (const pending of (cgPending[lifeline] || [])) {
    if (pending.id !== r.id && hasPath(pending, r)) return false;
  }
  return true;
}

function findReusableActRow(lifeline) {
  return Object.values(cgRows)
    .filter(r => canReuseActRow(lifeline, r))
    .sort((a, b) => b.level !== a.level ? b.level - a.level : b.id - a.id)[0] || null;
}

// r becomes the next materialized event on `lifeline`:
//   • it must follow the last event on that lifeline
//   • it must precede any still-pending receives on that lifeline
function materializeOnLifeline(lifeline, r) {
  cgAddEdge(cgLast[lifeline], r);
  for (const p of (cgPending[lifeline] || [])) {
    if (p.id !== r.id) cgAddEdge(r, p);
  }
  cgLast[lifeline] = r;
}

// Re-sort all row wrappers in the diagram by their level.
// Uses the "append each in order" trick: appendChild on an existing child moves it.
function syncOrder() {
  const sorted = Object.values(cgRows)
    .filter(r => r.wrapper)
    .sort((a, b) => a.level !== b.level ? a.level - b.level : a.id - b.id);
  for (const r of sorted) diagram.appendChild(r.wrapper);
  requestAnimationFrame(redrawAll);
}

// ── Replay button ──────────────────────────────────────────────────────
function showReplayButton() {
  const existing = document.getElementById('replay-btn');
  if (existing) return;
  const btn = document.createElement('button');
  btn.id = 'replay-btn';
  btn.textContent = '▶  Run again';
  btn.onclick = () => {
    btn.remove();
    statusEl.textContent = 'waiting…';
    statusEl.className = '';
    // POST resets bus history and publishes init before responding;
    // then reconnect so we pick up the clean history immediately.
    fetch('/replay', { method: 'POST' }).then(() => connect());
  };
  topbar.appendChild(btn);
}

// ── Act rows ───────────────────────────────────────────────────────────
function handleActStart(ev) {
  const i = lifelineIdx(ev.lifeline);
  const c = col(i);

  let r = findReusableActRow(ev.lifeline);
  if (r) {
    materializeOnLifeline(ev.lifeline, r);
  } else {
    r = newCgRow('action');
    const wrapper = document.createElement('div');
    wrapper.className = 'act-row';
    wrapper.style.gridTemplateColumns = `repeat(${N}, 1fr)`;
    r.wrapper = wrapper;
    r.cells = lifelines.map((_, k) => {
      const cell = document.createElement('div');
      cell.className = 'act-cell';
      cell.style.background = col(k).bg;
      wrapper.appendChild(cell);
      return cell;
    });
    materializeOnLifeline(ev.lifeline, r);
    diagram.appendChild(wrapper);
  }
  r.occupied.add(i);

  const box = document.createElement('div');
  box.className = 'act-box running';
  box.style.borderLeftColor = c.accent;

  const nameEl = document.createElement('div');
  nameEl.className = 'act-name';
  nameEl.style.color = c.text;
  nameEl.textContent = ev.action;
  box.appendChild(nameEl);

  const inStr = fmtDict(ev.inputs || {}, false);
  if (inStr) {
    box.appendChild(makeExpandableText('act-in', '← ' + inStr));
  }

  r.cells[i].appendChild(box);
  pendingActBoxes[`${ev.lifeline}:${ev.seq}`] = box;
  syncOrder();
  scrollDown();
}

function handleAct(ev) {
  const box = pendingActBoxes[`${ev.lifeline}:${ev.seq}`];
  if (!box) return;
  delete pendingActBoxes[`${ev.lifeline}:${ev.seq}`];

  box.classList.remove('running');

  const outStr = fmtDict(ev.outputs || {}, false);
  if (outStr) {
    box.appendChild(makeExpandableText('act-out', '→ ' + outStr));
  }
  scrollDown();
}

// ── Message rows ───────────────────────────────────────────────────────
function makeBox(label, text, color) {
  const box = document.createElement('div');
  box.className = 'msg-box';
  box.style.borderLeftColor = color;

  const lbl = document.createElement('div');
  lbl.className = 'msg-label';
  lbl.style.color = color;
  lbl.textContent = label;
  box.appendChild(lbl);

  if (text) {
    const vals = document.createElement('div');
    vals.className = 'msg-vals';
    vals.textContent = text;
    box.appendChild(vals);
  }
  return box;
}

function handleSend(ev) {
  const fromIdx = lifelineIdx(ev.from);
  const toIdx   = lifelineIdx(ev.to);
  const ctrl    = isCtrlVals(ev.values);
  const color   = col(fromIdx).accent;

  const r = newCgRow('message');
  r.fromIdx = fromIdx; r.toIdx = toIdx; r.ctrl = ctrl; r.color = color;

  const wrapper = document.createElement('div');
  wrapper.className = 'msg-row';
  wrapper.style.gridTemplateColumns = `repeat(${N}, 1fr)`;
  r.wrapper = wrapper;
  r.cells = lifelines.map((_, k) => {
    const cell = document.createElement('div');
    cell.className = 'msg-cell';
    cell.style.background = col(k).bg;
    wrapper.appendChild(cell);
    return cell;
  });

  r.cells[fromIdx].appendChild(makeBox('send', fmtList(ev.values) || null, color));
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  wrapper.appendChild(svg);
  r.svg = svg;

  // Register in the constraint graph: send event on from-lifeline
  materializeOnLifeline(ev.from, r);

  // Register as a pending receive on to-lifeline
  const chanKey = `${ev.from}->${ev.to}`;
  if (!cgOpen[chanKey]) cgOpen[chanKey] = [];
  cgOpen[chanKey].push(r);
  if (!cgPending[ev.to]) cgPending[ev.to] = new Set();
  cgPending[ev.to].add(r);

  // Register for arrow drawing now (recv box is empty until recv fires;
  // _drawArrow falls back to column-midpoint for the missing endpoint).
  const p = { row: wrapper, svg, cells: r.cells,
              fromIdx, toIdx, ctrl, color };
  r._completed = p;
  completed.push(p);

  diagram.appendChild(wrapper);
  syncOrder();
  scrollDown();
}

function handleRecv(ev) {
  const chanKey = `${ev.from}->${ev.to}`;
  const queue   = cgOpen[chanKey];
  if (!queue || !queue.length) return;
  const r = queue.shift();

  // Remove from pending-receive set
  cgPending[ev.to]?.delete(r);

  // Fill in recv box (arrow is redrawn by syncOrder → redrawAll)
  r.cells[r.toIdx].appendChild(
    makeBox('recv', fmtDict(ev.bindings || {}) || null, col(r.toIdx).accent));

  // Register recv event on to-lifeline (may raise level of later rows)
  materializeOnLifeline(ev.to, r);

  syncOrder();
  scrollDown();
}

// ── Arrow drawing ──────────────────────────────────────────────────────
function _drawArrow(p, dashed) {
  const row = p.row;
  const svg = p.svg;
  const W = row.clientWidth;
  const H = row.clientHeight || 56;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const sendBox  = p.cells[p.fromIdx].querySelector('.msg-box');
  const recvBox  = p.cells[p.toIdx].querySelector('.msg-box');
  const rowRect  = row.getBoundingClientRect();

  // getBoundingClientRect() returns visual pixels (affected by CSS zoom),
  // but clientWidth / viewBox are in CSS pixels.  Derive the zoom factor
  // from the row element itself so arrow coords stay in viewBox space.
  const zoom = W > 0 ? rowRect.width / W : 1;

  let x1, x2;
  const dir = p.toIdx > p.fromIdx ? 1 : -1;
  const cellW = W / N;

  if (sendBox) {
    const r = sendBox.getBoundingClientRect();
    x1 = (dir > 0 ? r.right - rowRect.left : r.left - rowRect.left) / zoom;
  } else {
    x1 = (p.fromIdx + 0.5) * cellW;
  }

  if (recvBox) {
    const r = recvBox.getBoundingClientRect();
    x2 = (dir > 0 ? r.left - rowRect.left : r.right - rowRect.left) / zoom;
  } else {
    x2 = (p.toIdx + 0.5) * cellW;
  }

  const y = H / 2;
  const color = p.color;

  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', x1); line.setAttribute('y1', y);
  line.setAttribute('x2', x2); line.setAttribute('y2', y);
  line.setAttribute('stroke', color);
  line.setAttribute('stroke-width', '2.4');
  if (dashed || p.ctrl) line.setAttribute('stroke-dasharray', '6 4');
  svg.appendChild(line);

  const arrowLen = 12;
  const arrowHalfHeight = 6.0;
  const pts = `${x2},${y} ${x2 - dir*arrowLen},${y - arrowHalfHeight} ${x2 - dir*arrowLen},${y + arrowHalfHeight}`;
  const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
  arrow.setAttribute('points', pts);
  arrow.setAttribute('fill', color);
  svg.appendChild(arrow);

  const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  dot.setAttribute('cx', x1); dot.setAttribute('cy', y); dot.setAttribute('r', '2.75');
  dot.setAttribute('fill', color);
  svg.appendChild(dot);
}

// ── Redraw all arrows on zoom / resize ────────────────────────────────
function redrawAll() {
  completed.forEach(p => { p.svg.innerHTML = ''; _drawArrow(p, false); });
}
new ResizeObserver(redrawAll).observe(container);

// ── SSE connection ─────────────────────────────────────────────────────
let eventSrc = null;
function connect() {
  if (eventSrc) { eventSrc.close(); eventSrc = null; }
  const src = new EventSource('/events');
  eventSrc = src;

  src.onopen = () => {
    statusEl.textContent = 'connected';
    statusEl.className = 'running';
  };

  src.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    switch (ev.type) {
      case 'init':
        clearDiagram();
        const rb = document.getElementById('replay-btn');
        if (rb) rb.remove();
        initDiagram(ev.lifelines);
        statusEl.textContent = 'running…';
        statusEl.className = 'running';
        break;
      case 'act_start': handleActStart(ev); break;
      case 'act':       handleAct(ev);      break;
      case 'send':      handleSend(ev);     break;
      case 'recv':      handleRecv(ev);     break;
      case 'done':
        actRow = null;
        statusEl.textContent = 'done ✓';
        statusEl.className = 'done';
        showReplayButton();
        break;
      case 'close':
        src.close();
        eventSrc = null;
        statusEl.textContent = 'stopped';
        statusEl.className = '';
        break;
    }
  };

  src.onerror = () => {
    statusEl.textContent = 'disconnected — retrying…';
    statusEl.className = '';
  };
}

connect();
</script>
</body>
</html>
"""
