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

For several independent workflow runs in the same page, use dashboard mode::

    wt = WebTrace.dashboard().start()
    first.configure(ui=True, trace=wt)
    second.configure(ui=True, trace=wt)
    first(...)
    second(...)
"""

from __future__ import annotations

import json
import pathlib
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_ASSETS = pathlib.Path(__file__).parent / "assets"

__all__ = ["WebTrace"]


def _lifeline_names(lifelines) -> list[str]:
    return [l.name if hasattr(l, "name") else str(l) for l in lifelines]


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
                  init_event: dict,
                  pending_human_inputs: dict):
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
            elif self.path == "/human-input":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode())
                req_id = body.get("id")
                raw_value = str(body.get("value", ""))
                if req_id and req_id in pending_human_inputs:
                    evt, result_box = pending_human_inputs[req_id]
                    result_box.append(raw_value)
                    evt.set()
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_response(404)
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

class _ScopedWebTrace:
    def __init__(self, parent: "WebTrace", name: str, lifelines):
        self._parent = parent
        self._run_id = uuid.uuid4().hex[:8]
        self._path = [self._run_id]
        self._lifelines = _lifeline_names(lifelines)
        parent._bus.publish({
            "type": "run_start",
            "run_id": self._run_id,
            "path": self._path,
            "name": name,
            "lifelines": self._lifelines,
        })

    @property
    def path(self) -> list[str]:
        return list(self._path)

    def __call__(self, event: dict) -> None:
        scoped = dict(event)
        scoped["path"] = self._path + list(event.get("path") or [])
        self._parent._bus.publish(scoped)

    def done(self) -> None:
        self._parent._bus.publish({"type": "done", "path": self.path})

    def make_human_backend(self):
        return self._parent._make_human_backend(path=self._path)


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

    def __init__(self, lifelines, port: int = 8765, *, dashboard: bool = False, name: str = ""):
        self._dashboard = dashboard
        self._lifelines = _lifeline_names(lifelines)
        self._port = port
        self._name = name
        self._bus = _EventBus()
        self._server: ThreadingHTTPServer | None = None
        self._replay_event = threading.Event()
        self._pending_human_inputs: dict[str, tuple[threading.Event, list]] = {}

    @classmethod
    def dashboard(cls, port: int = 8765) -> "WebTrace":
        """Create a ZipperChat dashboard for several independent workflow runs."""
        return cls([], port=port, dashboard=True)

    @property
    def is_dashboard(self) -> bool:
        return self._dashboard

    def _init_event(self) -> dict:
        if self._dashboard:
            return {"type": "init", "dashboard": True}
        ev: dict = {"type": "init", "lifelines": self._lifelines}
        if self._name:
            ev["name"] = self._name
        return ev

    def start(self) -> "WebTrace":
        if self._server is not None:
            return self

        init_ev = self._init_event()
        handler = _make_handler(
            self._bus, self._lifelines, self._replay_event, init_ev,
            self._pending_human_inputs,
        )
        self._server = _Server(("", self._port), handler)
        self._port = self._server.server_address[1]
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        print(f"ZipperChat → http://localhost:{self._port}")
        self._bus.publish(init_ev)
        return self

    def reset(self) -> "WebTrace":
        """Clear the diagram and prepare for a new run."""
        self._replay_event.clear()
        self._bus.reset()
        self._bus.publish(self._init_event())
        return self

    def wait_for_replay(self) -> None:
        """Block until the ▶ Run again button is clicked in the browser."""
        self._replay_event.wait()
        self._replay_event.clear()

    def __call__(self, event: dict) -> None:
        self._bus.publish(event)

    def done(self) -> None:
        self._bus.publish({"type": "done"})

    def start_run(self, name: str, lifelines) -> _ScopedWebTrace:
        """Start one workflow run inside a dashboard trace."""
        if not self._dashboard:
            raise RuntimeError("start_run() is only available on WebTrace.dashboard().")
        return _ScopedWebTrace(self, name, lifelines)

    def trace_run(self, name: str, lifelines) -> _ScopedWebTrace:
        """Alias for start_run(), kept close to the trace= use case."""
        return self.start_run(name, lifelines)

    def stop(self) -> None:
        self._bus.publish({"type": "close"})
        if self._server:
            self._server.shutdown()

    def _make_human_backend(self, path: list[str] | None = None):
        """Return a human backend callable that blocks until ZipperChat provides input."""
        pending = self._pending_human_inputs

        def backend(action, inputs: dict) -> dict:
            req_id = str(uuid.uuid4())
            evt = threading.Event()
            result_box: list = []
            pending[req_id] = (evt, result_box)

            prompt = action.prompt.format(**inputs)
            if action.output_type is bool:
                input_type = "bool"
            elif action.options is not None:
                input_type = "choice"
            else:
                input_type = "text"

            lifeline_name = threading.current_thread().name
            request_event: dict[str, object] = {
                "type": "human_input_required",
                "id": req_id,
                "lifeline": lifeline_name,
                "prompt": prompt,
                "input_type": input_type,
                "options": list(action.options) if action.options else None,
                "prefill": inputs[action.prefill] if action.prefill else None,
            }
            if path is not None:
                request_event["path"] = list(path)
            self._bus.publish(request_event)

            evt.wait()
            del pending[req_id]
            raw = result_box[0]

            if action.output_type is bool:
                value: object = str(raw).lower() in ("true", "yes", "1", "y")
            else:
                value = str(raw)

            input_event: dict[str, object] = {
                "type": "human_input",
                "id": req_id,
                "lifeline": lifeline_name,
                "value": str(value),
            }
            if path is not None:
                input_event["path"] = list(path)
            self._bus.publish(input_event)

            return {action.output: value}

        return backend

    def make_human_backend(self):
        return self._make_human_backend()


# ---------------------------------------------------------------------------
# Embedded viewer HTML
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZipperChat</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:        #F8F8F6;
  --bg-card:   #FFFFFF;
  --frame:     #143146;
  --frame-fg:  #F2EDE6;
  --ink:       #1A1A1A;
  --ink-mute:  #6B7280;
  --ink-faint: #9CA3AF;
  --hair:      #E5E7EB;
  --hair-s:    #D1D5DB;
  --k-llm:     #6366F1;
  --k-llm-bg:  #EEF2FF;
  --k-pure:    #6B7280;
  --k-pure-bg: #F3F4F6;
  --k-human:   #C2495A;
  --k-human-bg:#FDF2F3;
  --k-plan:    #D97706;
  --k-plan-bg: #FFFBEB;
  --radius:    8px;
}
body.dark {
  --bg:        #0D1B2A;
  --bg-card:   #142032;
  --ink:       #E8E6E3;
  --ink-mute:  #8B94A3;
  --ink-faint: #3D4F63;
  --hair:      #1C2E42;
  --hair-s:    #253D56;
  --k-llm-bg:  #11183A;
  --k-pure-bg: #1A2535;
  --k-human-bg:#200E12;
  --k-plan-bg: #1F1408;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--ink); font-size: 13px; }

/* ── Layout ────────────────────────────────────────────────────────────── */
#app  { display: flex; flex-direction: column; height: 100%; }
#main { display: flex; flex: 1; overflow: hidden; }

/* ── Header ────────────────────────────────────────────────────────────── */
#hdr {
  display: flex; align-items: center; gap: 10px;
  padding: 0 16px; height: 48px;
  background: var(--frame); color: var(--frame-fg);
  flex-shrink: 0;
}
.hdr-logo { font-size: 12px; font-weight: 700; letter-spacing: 0.06em; opacity: 0.85; }
.hdr-sep  { width: 1px; height: 18px; background: rgba(242,237,230,.2); margin: 0 2px; }
#wf-name  { font-size: 13px; font-weight: 500; opacity: 0.8; }
.hdr-gap  { flex: 1; }
.s-dot {
  width: 7px; height: 7px; border-radius: 50%; background: #4B5563;
  transition: background .3s;
}
.s-dot.running { background: #4ADE80; }
.s-dot.waiting { background: #FBBF24; animation: pulse 1.2s ease-in-out infinite; }
.s-dot.done    { background: #818CF8; }
.s-dot.error   { background: #F87171; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.45} }
#s-label { font-size: 11px; color: rgba(242,237,230,.55); letter-spacing: .02em; }
#dark-btn {
  background: none; border: 1px solid rgba(242,237,230,.2); color: var(--frame-fg);
  cursor: pointer; width: 28px; height: 28px; border-radius: 6px;
  font-size: 13px; display: flex; align-items: center; justify-content: center;
  transition: border-color .15s;
}
#dark-btn:hover { border-color: rgba(242,237,230,.55); }

/* ── Timeline (left) ───────────────────────────────────────────────────── */
#timeline {
  flex: 1; overflow-x: auto; overflow-y: hidden;
  display: flex; gap: 1px; background: var(--hair);
  align-items: stretch; min-width: 0;
}
.ll-col {
  display: flex; flex-direction: column;
  min-width: 160px; flex: 1; max-width: 240px;
  background: var(--bg); overflow: hidden;
}
.ll-header {
  padding: 7px 10px; flex-shrink: 0;
  background: var(--bg-card); border-bottom: 1px solid var(--hair);
  font-size: 10px; font-weight: 700; letter-spacing: .07em;
  color: var(--ink-mute); text-transform: uppercase;
}
.ll-body {
  flex: 1; overflow-y: auto; padding: 6px 5px;
  display: flex; flex-direction: column; gap: 3px;
}
/* ── Action card ───────────────────────────────────────────────────────── */
.ac {
  border-radius: 5px; padding: 5px 7px;
  border: 1px solid var(--hair); background: var(--bg-card);
  transition: opacity .2s;
}
.ac.pending { opacity: .6; }
.ac-top { display: flex; align-items: center; gap: 5px; }
.ac-kind {
  font-size: 7.5px; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; padding: 1px 4px; border-radius: 3px; flex-shrink: 0;
}
.k-llm    { background: var(--k-llm-bg);   color: var(--k-llm); }
.k-pure   { background: var(--k-pure-bg);  color: var(--k-pure); }
.k-human  { background: var(--k-human-bg); color: var(--k-human); }
.k-planner{ background: var(--k-plan-bg);  color: var(--k-plan); }
.ac-name {
  font-size: 11px; font-weight: 500; color: var(--ink);
  flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.ac-st { font-size: 10px; color: var(--ink-faint); flex-shrink: 0; }
.ac-st.run { color: #4ADE80; animation: pulse 1s ease-in-out infinite; }
.ac-st.err { color: #F87171; }
.ac-outs {
  margin-top: 3px; font-size: 10px;
  font-family: 'JetBrains Mono', monospace; color: var(--ink-mute); line-height: 1.4;
}
.ac-out { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
/* send/recv tag */
.msg-tag {
  font-size: 9px; color: var(--ink-faint); padding: 1px 0 0 1px;
  font-family: 'JetBrains Mono', monospace; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}

/* ── Inspector (right) ─────────────────────────────────────────────────── */
#inspector {
  width: 380px; flex-shrink: 0;
  display: flex; flex-direction: column;
  border-left: 1px solid var(--hair); background: var(--bg-card);
  overflow: hidden;
}
#ins-hdr {
  display: flex; align-items: center; gap: 8px;
  padding: 0 14px; height: 40px; flex-shrink: 0;
  border-bottom: 1px solid var(--hair);
  font-size: 10px; font-weight: 700; letter-spacing: .07em;
  color: var(--ink-mute); text-transform: uppercase;
}
#q-count {
  background: var(--k-human); color: #fff;
  font-size: 9px; font-weight: 700; border-radius: 10px;
  padding: 1px 7px; display: none;
}
#q-count.on { display: inline-block; }
/* ins-body fills the panel; its children define the layout per view */
#ins-body {
  flex: 1; min-height: 0;
  display: flex; flex-direction: column;
  overflow: hidden;
}
.ins-empty {
  flex: 1; display: flex; align-items: center; justify-content: center;
  font-size: 12px; color: var(--ink-faint); text-align: center; padding: 24px;
}
/* Shared section header */
.ins-sec-hdr {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  padding: 8px 14px; flex-shrink: 0;
  background: var(--bg-card); border-bottom: 1px solid var(--hair);
}
.ins-ll {
  font-size: 9px; font-weight: 700; letter-spacing: .05em;
  text-transform: uppercase; padding: 2px 7px; border-radius: 4px;
  background: var(--k-human-bg); color: var(--k-human);
}
.ins-title { font-size: 12px; font-weight: 500; color: var(--ink); }
.ins-badge { font-size: 10px; color: var(--ink-faint); margin-left: auto; }
/* Prompt: scrollable, fills remaining height */
.ins-prompt {
  flex: 1; overflow-y: auto;
  padding: 12px 14px; font-size: 12px; line-height: 1.65;
  white-space: pre-wrap; font-family: 'JetBrains Mono', monospace; color: var(--ink);
}
/* Widget: pinned at bottom, always visible */
.ins-widget {
  flex-shrink: 0; padding: 12px 14px;
  border-top: 1px solid var(--hair); background: var(--bg-card);
}
.ins-resolved {
  flex-shrink: 0; padding: 10px 14px;
  font-size: 11px; font-family: 'JetBrains Mono', monospace;
  color: var(--ink-mute); border-top: 1px solid var(--hair); background: var(--bg-card);
}
/* Action outputs (card detail) */
.ins-outputs {
  flex: 1; overflow-y: auto; padding: 12px 14px;
}
.ins-out-row { margin-bottom: 10px; }
.ins-out-key { font-size: 10px; color: var(--ink-faint); font-family: 'JetBrains Mono', monospace; }
.ins-out-val {
  font-size: 12px; font-family: 'JetBrains Mono', monospace; color: var(--ink);
  white-space: pre-wrap; word-break: break-word; line-height: 1.5; margin-top: 2px;
}
/* Timeline card: hoverable, selectable */
.ac { cursor: pointer; }
.ac:hover { border-color: var(--hair-s); }
.ac.selected      { border-color: var(--k-llm) !important; }
.ac.human-pending { border-color: var(--k-human) !important; opacity: 1; }
/* Input widgets */
.qi-ta {
  width: 100%; font-size: 12px; font-family: 'JetBrains Mono', monospace;
  min-height: 72px; padding: 7px 9px; resize: vertical;
  border-radius: 5px; border: 1px solid var(--hair-s);
  background: var(--bg-card); color: var(--ink);
  outline: none; transition: border-color .15s; box-sizing: border-box;
}
.qi-ta::placeholder { color: var(--ink-faint); }
.qi-ta:focus { border-color: var(--k-human); }
.qi-submit-row { display: flex; justify-content: flex-end; margin-top: 8px; }
.qi-btn {
  font-size: 12px; font-weight: 600; padding: 7px 20px;
  border-radius: 5px; border: 1px solid var(--k-human);
  background: var(--k-human-bg); color: var(--k-human);
  cursor: pointer; transition: background .12s, opacity .12s;
}
.qi-btn:hover:not(:disabled) { background: rgba(194,73,90,.12); }
.qi-btn:disabled { opacity: .3; cursor: default; }
.qi-bool-row { display: flex; gap: 8px; }
.qi-bool {
  flex: 1; font-size: 13px; font-weight: 600; padding: 10px 12px;
  border-radius: 5px; border: 1px solid var(--hair-s);
  background: var(--bg-card); color: var(--ink);
  cursor: pointer; transition: background .12s, border-color .12s;
}
.qi-bool:hover:not(:disabled) { border-color: var(--ink-mute); background: var(--bg); }
.qi-bool.yes:hover:not(:disabled) { background: #F0FDF4; border-color: #4ADE80; color: #166534; }
.qi-bool.no:hover:not(:disabled)  { background: #FFF1F2; border-color: #F87171; color: #991B1B; }
.qi-bool:disabled { opacity: .3; cursor: default; }
body.dark .qi-bool.yes:hover:not(:disabled) { background:#052E16; border-color:#166534; color:#4ADE80; }
body.dark .qi-bool.no:hover:not(:disabled)  { background:#2D0708; border-color:#991B1B; color:#F87171; }
/* Scrollbars */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--hair-s); border-radius: 3px; }
</style>
</head>
<body>
<div id="app">
  <header id="hdr">
    <span class="hdr-logo">ZIPPERCHAT</span>
    <div class="hdr-sep"></div>
    <span id="wf-name">—</span>
    <div class="hdr-gap"></div>
    <span class="s-dot" id="s-dot"></span>
    <span id="s-label">connecting…</span>
    <button id="dark-btn" title="Toggle dark mode">🌙</button>
  </header>
  <div id="main">
    <div id="timeline">
      <p style="padding:20px;color:var(--ink-faint);font-size:12px;">Awaiting workflow…</p>
    </div>
    <div id="inspector">
      <div id="ins-hdr">
        Inspector
        <span id="q-count"></span>
      </div>
      <div id="ins-body">
        <div class="ins-empty">Click an action to inspect it</div>
      </div>
    </div>
  </div>
</div>
<script>
// ── State ──────────────────────────────────────────────────────────────────
let evSrc      = null;
const cols     = {};        // lifeline name → {bodyEl}
const cards    = {};        // `${ll}:${seq}` → {el, outsEl, kind, name, lifeline, outputs, reqId}
const reqMap   = new Map(); // req_id → {lifeline, prompt, input_type, options, prefill, resolved, value}
let pending    = 0;
let selReqId   = null;      // req_id currently shown in inspector
let selCardKey = null;      // card key currently shown in inspector

// ── DOM ────────────────────────────────────────────────────────────────────
const timeline = document.getElementById('timeline');
const insBody  = document.getElementById('ins-body');
const qCount   = document.getElementById('q-count');
const sDot     = document.getElementById('s-dot');
const sLabel   = document.getElementById('s-label');
const wfName   = document.getElementById('wf-name');
const darkBtn  = document.getElementById('dark-btn');

// ── Dark mode ──────────────────────────────────────────────────────────────
(()=>{ if(localStorage.getItem('zc-theme')==='dark') document.body.classList.add('dark'); })();
darkBtn.addEventListener('click', ()=>{
  document.body.classList.toggle('dark');
  localStorage.setItem('zc-theme', document.body.classList.contains('dark')?'dark':'light');
});

// ── Helpers ────────────────────────────────────────────────────────────────
function esc(s){ return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function fmtV(v){
  if(v===null||v===undefined) return '';
  if(typeof v==='string'&&v.startsWith('κ_ctrl_')) return '';
  if(typeof v==='boolean') return v?'true':'false';
  if(typeof v==='string') return v.length>400?v.slice(0,399)+'…':v;
  try{ return JSON.stringify(v,null,2); }catch{ return String(v); }
}
function setStatus(cls, label){
  sDot.className = 's-dot '+cls;
  sLabel.textContent = label;
}
function refreshCount(){
  if(pending>0){ qCount.textContent=pending; qCount.classList.add('on'); }
  else{ qCount.classList.remove('on'); }
}
function clearSelected(){
  document.querySelectorAll('.ac.selected').forEach(el=>el.classList.remove('selected'));
}

// ── Inspector: show human req ──────────────────────────────────────────────
function showReq(req_id){
  const req = reqMap.get(req_id);
  if(!req) return;
  selReqId = req_id; selCardKey = null;
  clearSelected();
  for(const c of Object.values(cards)){ if(c.reqId===req_id) c.el.classList.add('selected'); }

  const isBool   = req.input_type==='bool';
  const isChoice = req.input_type==='choice';

  let widgetHtml = '';
  if(req.resolved){
    const shown = req.value===''||req.value==='(approved as-is)' ? 'approved as-is' : req.value;
    widgetHtml = `<div class="ins-resolved">✓ ${esc(shown)}</div>`;
  } else if(isBool){
    widgetHtml = `<div class="ins-widget"><div class="qi-bool-row">
      <button class="qi-bool yes" disabled>Yes</button>
      <button class="qi-bool no"  disabled>No</button>
    </div></div>`;
  } else if(isChoice&&req.options){
    const btns = req.options.map(o=>`<button class="qi-bool" disabled data-val="${esc(o)}">${esc(o)}</button>`).join('');
    widgetHtml = `<div class="ins-widget"><div class="qi-bool-row">${btns}</div></div>`;
  } else {
    const prefillVal = req.prefill ? esc(req.prefill) : '';
    widgetHtml = `<div class="ins-widget">
      <textarea class="qi-ta" rows="3">${prefillVal}</textarea>
      <div class="qi-submit-row"><button class="qi-btn" disabled>Submit →</button></div>
    </div>`;
  }

  insBody.innerHTML = `
    <div class="ins-sec-hdr">
      <span class="ins-ll">${esc(req.lifeline)}</span>
      <span class="ins-title">${isBool?'Decision required':'Input required'}</span>
      ${req.resolved?'<span class="ins-badge">✓ done</span>':''}
    </div>
    <div class="ins-prompt">${esc(req.prompt)}</div>
    ${widgetHtml}`;

  if(!req.resolved){
    if(isBool){
      const yes=insBody.querySelector('.yes'), no=insBody.querySelector('.no');
      setTimeout(()=>{ yes.disabled=false; no.disabled=false; },600);
      yes.onclick=()=>doSubmit(req_id,'true');
      no.onclick =()=>doSubmit(req_id,'false');
    } else if(isChoice){
      const btns=insBody.querySelectorAll('.qi-bool');
      setTimeout(()=>btns.forEach(b=>b.disabled=false),600);
      btns.forEach(b=>{ b.onclick=()=>doSubmit(req_id,b.dataset.val); });
    } else {
      const ta=insBody.querySelector('.qi-ta'), btn=insBody.querySelector('.qi-btn');
      setTimeout(()=>btn.disabled=false,800);
      setTimeout(()=>ta.focus({preventScroll:true}),900);
      btn.onclick=()=>doSubmit(req_id,ta.value);
    }
  }
}

// ── Inspector: show card detail ────────────────────────────────────────────
function showCard(key){
  const c=cards[key];
  if(!c){ return; }
  if(c.reqId && !reqMap.get(c.reqId)?.resolved){ showReq(c.reqId); return; }
  selCardKey=key; selReqId=null;
  clearSelected(); c.el.classList.add('selected');

  const outEntries=Object.entries(c.outputs||{})
    .filter(([,v])=>!(typeof v==='string'&&v.startsWith('κ_ctrl_')));
  const outHtml=outEntries.length
    ? outEntries.map(([k,v])=>`<div class="ins-out-row">
        <div class="ins-out-key">${esc(k)}</div>
        <div class="ins-out-val">${esc(fmtV(v))}</div>
      </div>`).join('')
    : '<p style="color:var(--ink-faint);font-size:12px">No outputs yet</p>';

  insBody.innerHTML = `
    <div class="ins-sec-hdr">
      <span class="ac-kind k-${esc(c.kind)}">${esc(kindLabel(c.kind))}</span>
      <span class="ins-title">${esc(c.name)}</span>
      <span class="ins-badge">${esc(c.lifeline)}</span>
    </div>
    <div class="ins-outputs">${outHtml}</div>`;
}

// ── Submit ─────────────────────────────────────────────────────────────────
function doSubmit(req_id, val){
  fetch('/human-input',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:req_id,value:val})});
}

// ── Lifeline columns ───────────────────────────────────────────────────────
function ensureCol(name){
  if(cols[name]) return;
  const col=document.createElement('div');
  col.className='ll-col';
  col.innerHTML=`<div class="ll-header">${esc(name)}</div><div class="ll-body" id="llb-${esc(name)}"></div>`;
  timeline.appendChild(col);
  cols[name]={ bodyEl: col.querySelector('.ll-body') };
}

const KIND_LABEL={llm:'LLM',pure:'PURE',human:'HUMAN',planner:'PLAN'};
function kindLabel(k){ return KIND_LABEL[k]||'ACT'; }

// ── Event handlers ─────────────────────────────────────────────────────────
function handleInit(e){
  timeline.innerHTML='';
  Object.keys(cols).forEach(k=>delete cols[k]);
  Object.keys(cards).forEach(k=>delete cards[k]);
  reqMap.clear(); pending=0; selReqId=null; selCardKey=null;
  wfName.textContent=e.name||'workflow';
  setStatus('running','running');
  (e.lifelines||[]).forEach(ll=>ensureCol(ll));
  insBody.innerHTML='<div class="ins-empty">Click an action to inspect it</div>';
  refreshCount();
}

function handleActStart(e){
  const ll=e.lifeline;
  if(!cols[ll]) ensureCol(ll);
  const body=cols[ll].bodyEl;
  const kind=e.action_kind||'pure';
  const name=e.action||'—';
  const key=ll+':'+e.seq;
  if(name==='assign') return;

  const card=document.createElement('div');
  card.className='ac pending';
  card.innerHTML=`
    <div class="ac-top">
      <span class="ac-kind k-${esc(kind)}">${esc(kindLabel(kind))}</span>
      <span class="ac-name" title="${esc(name)}">${esc(name)}</span>
      <span class="ac-st run">●</span>
    </div>
    <div class="ac-outs"></div>`;
  body.appendChild(card);
  body.scrollTop=body.scrollHeight;
  cards[key]={ el:card, outsEl:card.querySelector('.ac-outs'), kind, name, lifeline:ll, outputs:{}, reqId:null };
  card.onclick=()=>showCard(key);
}

function handleAct(e){
  const key=e.lifeline+':'+e.seq;
  const c=cards[key]; if(!c) return;
  c.el.classList.remove('pending','human-pending');
  const st=c.el.querySelector('.ac-st');
  st.className='ac-st'; st.textContent='✓';
  c.outputs=e.outputs||{};
  const lines=Object.entries(e.outputs||{})
    .filter(([,v])=>!(typeof v==='string'&&v.startsWith('κ_ctrl_')))
    .map(([k,v])=>{ const val=fmtV(v); return val?`<div class="ac-out"><span style="color:var(--ink-faint)">${esc(k)} </span>${esc(val)}</div>`:''; }).join('');
  if(lines) c.outsEl.innerHTML=lines;
  if(selCardKey===key) showCard(key);
}

function handleSend(e){
  const ll=e.from||e.lifeline; if(!cols[ll]) return;
  const body=cols[ll].bodyEl;
  const last=body.querySelector('.ac:last-child'); if(!last) return;
  const tag=document.createElement('div');
  tag.className='msg-tag'; tag.textContent='→ '+(e.to||'');
  last.appendChild(tag);
}

function handleHumanRequired(e){
  pending++;
  setStatus('waiting','your turn');
  refreshCount();
  reqMap.set(e.id,{ lifeline:e.lifeline, prompt:e.prompt||'', input_type:e.input_type, options:e.options, prefill:e.prefill||null, resolved:false, value:null });

  // Link to most recent pending human card for this lifeline
  if(cols[e.lifeline]){
    const allCards=Array.from(cols[e.lifeline].bodyEl.querySelectorAll('.ac.pending'));
    for(let i=allCards.length-1;i>=0;i--){
      if(allCards[i].querySelector('.k-human')){
        allCards[i].classList.add('human-pending');
        for(const [k,c] of Object.entries(cards)){ if(c.el===allCards[i]){ c.reqId=e.id; break; } }
        break;
      }
    }
  }
  showReq(e.id);
}

function handleHumanInput(e){
  const req=reqMap.get(e.id);
  if(req){ req.resolved=true; req.value=e.value; }
  pending=Math.max(0,pending-1);
  if(pending===0) setStatus('running','running');
  refreshCount();
  for(const c of Object.values(cards)){ if(c.reqId===e.id){ c.el.classList.remove('human-pending'); break; } }
  if(selReqId===e.id) showReq(e.id);
  // Auto-advance to next pending req
  for(const [id,r] of reqMap){ if(!r.resolved){ setTimeout(()=>showReq(id),350); return; } }
}

function handleDone(){
  if(pending===0) setStatus('done','done');
}

function handleRunStart(e){
  (e.lifelines||[]).forEach(ll=>ensureCol(ll));
}

// ── Dispatcher ─────────────────────────────────────────────────────────────
function dispatch(e){
  if(!e||!e.type) return;
  switch(e.type){
    case 'init':                 handleInit(e);          break;
    case 'run_start':            handleRunStart(e);      break;
    case 'act_start':            handleActStart(e);      break;
    case 'act':                  handleAct(e);           break;
    case 'send':                 handleSend(e);          break;
    case 'human_input_required': handleHumanRequired(e); break;
    case 'human_input':          handleHumanInput(e);    break;
    case 'done':                 handleDone();           break;
    case 'error':                setStatus('error','error'); break;
  }
}

// ── SSE ────────────────────────────────────────────────────────────────────
function connect(){
  if(evSrc){ evSrc.close(); evSrc=null; }
  const src=new EventSource('/events');
  evSrc=src;
  src.onopen    =()=>setStatus('','connected');
  src.onmessage =ev=>{ try{ dispatch(JSON.parse(ev.data)); }catch(err){ console.error(err,ev.data); } };
  src.onerror   =()=>setStatus('error','reconnecting…');
}
connect();
</script>
</body>
</html>
"""
