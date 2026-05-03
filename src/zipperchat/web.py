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
<title>ZipperChat</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/* ─── Variables (navy chrome + cream stage + sherbet event bars) ─────────── */
:root {
  /* Surfaces */
  --bg:            #F8F8F6;            /* warm gray-white page (the "stage") */
  --bg-card:       #FFFFFF;            /* drawer + card surface              */
  --bg-hover:      #EDECEA;            /* hover wash on gray-white           */

  /* Frame (navy chrome) */
  --frame:         #143146;            /* deep navy chrome                   */
  --frame-fg:      #F2EDE6;            /* text on navy                       */
  --frame-mute:    #9CB0C2;            /* muted text on navy                 */

  /* Ink — single dark text color */
  --ink:           #143146;
  --ink-mute:      #4D6479;
  --ink-faint:     #8195A6;
  --ink-ghost:     #BCC8D2;

  /* Hairlines */
  --hairline:        rgba(20,49,70,0.10);
  --hairline-strong: rgba(20,49,70,0.20);
  --hairline-frame:  rgba(255,255,255,0.10);

  /* Agent identity (lifeline rails — quiet blue→gray) */
  --a-1: #3F5A78;
  --a-2: #5A7390;
  --a-3: #6E8298;
  --a-4: #7E8B9C;
  --a-5: #8E96A2;

  /* Event-kind fills (sherbet) — bar fills, calibrated for navy ink */
  --k-action:    #69A6E0;   /* periwinkle — actions / send-only           */
  --k-model:     #9C8FD9;   /* lavender — planner / workflow-spawning act */
  --k-tool:      #F1B07A;   /* peach — (reserved)                          */
  --k-send:      #7FB1B0;   /* teal — outbound messages                   */
  --k-recv:      #BFD9D8;   /* light teal — inbound (lighter sibling)     */
  --k-decision:  #E06B8A;   /* hot pink — if/while diamonds                */
  --k-ctrl:      #C4CCD4;   /* muted gray — control messages              */

  /* Functional status (single-color semantics across the UI) */
  --status-run:   var(--k-action);    /* periwinkle = running               */
  --status-done:  var(--k-send);      /* teal = done                        */
  --status-err:   var(--k-decision);  /* pink = error                       */
  --status-idle:  var(--ink-ghost);
}

/* ─── Dark theme ─────────────────────────────────────────────────────────── */
body.dark {
  --bg:              #0E1B2A;
  --bg-card:         #14283C;
  --bg-hover:        #1C3250;
  --frame:           #07111B;
  --frame-fg:        #E8F0F8;
  --frame-mute:      #6A8EA8;
  --ink:             #DDE8F0;
  --ink-mute:        #8AAEC8;
  --ink-faint:       #5A7A96;
  --ink-ghost:       #2E4860;
  --hairline:        rgba(255,255,255,0.07);
  --hairline-strong: rgba(255,255,255,0.13);
  --hairline-frame:  rgba(255,255,255,0.06);
}

/* ─── Reset & base ───────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', system-ui, sans-serif;
  font-size: 13px;
  line-height: 1.45;
  letter-spacing: -0.005em;
  background: var(--bg);
  color: var(--ink);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
button { font-family: inherit; }
.mono { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace; }
::selection { background: rgba(105, 166, 224, 0.32); color: var(--ink); }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--hairline-strong); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--ink-faint); }
::-webkit-scrollbar-corner { background: transparent; }

/* ─── Topbar (navy chrome) ──────────────────────────────────────────────── */
#topbar {
  height: 52px;
  flex-shrink: 0;
  background: var(--frame);
  color: var(--frame-fg);
  display: flex;
  align-items: stretch;
  border-bottom: 1px solid #0B1A28;
}
.brand {
  display: flex;
  align-items: center;
  padding: 0 20px;
  gap: 11px;
  border-right: 1px solid var(--hairline-frame);
}
.brand-mark {
  width: 12px;
  height: 12px;
  background: var(--k-action);
  border-radius: 2px;
  flex-shrink: 0;
}
.brand-name {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--frame-fg);
}
.brand-name-accent { color: var(--k-action); }

#topbar-meta {
  display: flex;
  align-items: center;
  flex: 1;
  padding: 0 18px;
  min-width: 0;
}

#status-indicator {
  display: flex;
  align-items: center;
  gap: 9px;
  margin-left: auto;
  padding: 4px 12px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--frame-mute);
  flex-shrink: 0;
  border-radius: 999px;
  background: rgba(255,255,255,0.04);
  transition: background 0.15s ease;
}
#status-indicator.connected,
#status-indicator.running { color: var(--frame-fg); background: rgba(105,166,224,0.10); }
#status-indicator.done    { color: var(--frame-fg); background: rgba(127,177,176,0.16); }
#status-indicator.error   { color: var(--frame-fg); background: rgba(224,107,138,0.16); }
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--status-idle);
  flex-shrink: 0;
  transition: background 0.2s ease, box-shadow 0.2s ease;
}
.status-dot.connected,
.status-dot.running {
  background: var(--status-run);
  box-shadow: 0 0 0 3px rgba(105,166,224,0.20);
  animation: status-pulse 1.6s ease-in-out infinite;
}
.status-dot.done {
  background: var(--status-done);
  box-shadow: 0 0 0 3px rgba(127,177,176,0.18);
}
.status-dot.error {
  background: var(--status-err);
  box-shadow: 0 0 0 3px rgba(224,107,138,0.18);
}
@keyframes status-pulse {
  0%, 100% { box-shadow: 0 0 0 2px rgba(105,166,224,0.20); }
  50%      { box-shadow: 0 0 0 5px rgba(105,166,224,0.10); }
}

/* ─── Dark-mode toggle ───────────────────────────────────────────────────── */
#dark-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  align-self: center;
  width: 32px;
  height: 32px;
  margin: 0 10px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: var(--frame-mute);
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
  flex-shrink: 0;
}
#dark-toggle:hover { background: rgba(255,255,255,0.08); color: var(--frame-fg); }
#dark-toggle .icon-sun  { display: block; }
#dark-toggle .icon-moon { display: none;  }
body.dark #dark-toggle .icon-sun  { display: none;  }
body.dark #dark-toggle .icon-moon { display: block; }

/* Event boxes keep dark text in dark mode (sherbet fills are light) */
body.dark .ev-box { --ink: #143146; --ink-mute: #4D6479; --ink-faint: #8195A6; }

/* ─── Body ───────────────────────────────────────────────────────────────── */
#body {
  flex: 1;
  display: flex;
  min-height: 0;
}

/* ─── Sidebar ───────────────────────────────────────────────────────────── */
#sidebar {
  width: 232px;
  flex-shrink: 0;
  background: var(--bg-card);
  border-right: 1px solid var(--hairline);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.sidebar-section {
  padding: 18px 16px 12px;
  flex: 1;
  overflow-y: auto;
}
.section-label {
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin-bottom: 14px;
  padding: 0 4px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--hairline);
}

/* Workflow tree */
#wf-tree {
  list-style: none;
}
.wf-tree-children {
  list-style: none;
  margin-left: 13px;
  padding-left: 11px;
  border-left: 1px solid var(--hairline);
}
.wf-tree-row {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 6px 8px;
  cursor: pointer;
  border: 1px solid transparent;
  border-radius: 6px;
  transition: background 0.12s ease, border-color 0.12s ease;
  position: relative;
  margin: 1px 0;
}
.wf-tree-row:hover {
  background: var(--bg-hover);
}
.wf-tree-row.focused {
  background: var(--bg-hover);
  border-color: var(--hairline-strong);
}
.wf-tree-status {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  background: var(--status-idle);
  transition: background 0.2s ease, box-shadow 0.2s ease;
}
.wf-tree-status.running {
  background: var(--status-run);
  box-shadow: 0 0 0 3px rgba(105,166,224,0.16);
  animation: status-pulse 1.6s ease-in-out infinite;
}
.wf-tree-status.done {
  background: var(--status-done);
  box-shadow: 0 0 0 2px rgba(127,177,176,0.16);
}
.wf-tree-name {
  font-size: 12.5px;
  color: var(--ink);
  font-weight: 500;
  letter-spacing: -0.005em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}
.wf-tree-row.running .wf-tree-name { font-weight: 600; }

.sidebar-bottom {
  padding: 14px 16px;
  border-top: 1px solid var(--hairline);
  flex-shrink: 0;
  background: var(--bg-card);
}
#replay-btn {
  width: 100%;
  height: 36px;
  padding: 0 14px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.01em;
  color: var(--ink-ghost);
  background: var(--bg);
  border: 1px solid var(--hairline);
  border-radius: 8px;
  cursor: not-allowed;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
}
#replay-btn:not([disabled]) {
  color: var(--frame);
  border-color: var(--ink);
  background: var(--bg-card);
  cursor: pointer;
}
#replay-btn:not([disabled]):hover {
  background: var(--ink);
  color: var(--frame-fg);
  box-shadow: 0 1px 6px rgba(20,49,70,0.18);
}
#replay-btn .btn-icon { font-size: 9px; }

/* Replay button dark-mode overrides */
body.dark #replay-btn              { color: var(--ink-faint); }
body.dark #replay-btn:not([disabled]) {
  color: var(--frame-fg);
  border-color: var(--hairline-strong);
  background: rgba(255,255,255,0.06);
}
body.dark #replay-btn:not([disabled]):hover {
  background: var(--frame-fg);
  color: var(--frame);
  box-shadow: 0 1px 6px rgba(0,0,0,0.30);
}

/* ─── Main diagram area (the cream stage) ───────────────────────────────── */
#main {
  flex: 1;
  min-width: 0;
  background: var(--bg);
  position: relative;
  overflow: hidden;
}
#diagram-root {
  position: absolute;
  inset: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  padding: 22px 24px 60px;
  gap: 18px;
}
.empty-msg {
  margin: auto;
  text-align: center;
  color: var(--ink-faint);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 60px 28px;
  border: 1px dashed var(--hairline);
  background: var(--bg-card);
  border-radius: 10px;
}

/* ─── Workflow group (a "level" — white card on cream) ──────────────────── */
.wf-group {
  display: flex;
  flex-direction: column;
  position: relative;
  --wf-accent: var(--k-model);
}
/* Active workflow: subtle left border in the level's accent color */
.wf-group.wf-active {
  border-radius: 10px;
  box-shadow: -3px 0 0 0 var(--spawner-color, rgba(156,143,217,0.4));
}
/* Children container */
.wf-children {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px 0 0 36px;
}
.wf-children:empty { display: none; }
/* All group labels are clickable */
.wf-label { cursor: pointer; }
.wf-label {
  display: flex;
  align-items: center;
  gap: 11px;
  font-size: 11px;
  font-weight: 600;
  color: var(--ink);
  padding: 9px 14px;
  background: var(--bg-card);
  border: 1px solid var(--hairline);
  border-bottom: none;
  border-radius: 10px 10px 0 0;
  letter-spacing: -0.005em;
}
.wf-label-name {
  font-weight: 600;
  font-size: 12.5px;
  color: var(--ink);
}
.wf-label-lifelines {
  font-size: 10.5px;
  font-weight: 500;
  color: var(--ink-mute);
  letter-spacing: 0.01em;
}
.wf-label-context {
  margin-left: auto;
  color: var(--ink-faint);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: -0.005em;
}
.wf-label-context strong {
  color: var(--ink-mute);
  font-weight: 600;
}

.wf-body {
  display: flex;
  background: var(--bg-card);
  border: 1px solid var(--hairline);
  border-radius: 0 0 10px 10px;
  min-height: 80px;
  overflow: hidden;
}

/* Frozen lifeline strip (white, no rail through it) */
.ll-strip {
  width: 152px;
  flex-shrink: 0;
  background: var(--bg-card);
  border-right: 1px solid var(--hairline);
  display: flex;
  flex-direction: column;
  z-index: 2;
}
.ll-strip-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 14px;
  height: var(--row-h, 76px);
  position: relative;
  --ll-color: var(--ink-faint);
  transition: background 0.18s ease;
}
.ll-strip-swatch {
  width: 4px;
  height: 22px;
  background: var(--ll-color);
  border-radius: 2px;
  flex-shrink: 0;
}
.ll-strip-name {
  font-size: 12.5px;
  font-weight: 600;
  color: var(--ink);
  letter-spacing: -0.005em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}
.ll-strip-row.thinking .ll-strip-name { color: var(--ink); }
.ll-strip-status {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: transparent;
  flex-shrink: 0;
  transition: background 0.2s ease, box-shadow 0.2s ease;
}
.ll-strip-row.thinking .ll-strip-status {
  background: var(--status-run);
  box-shadow: 0 0 0 3px rgba(105,166,224,0.20);
  animation: status-pulse 1.2s ease-in-out infinite;
}
.ll-strip-row.done .ll-strip-status {
  background: var(--status-done);
}

/* Scrollable events */
.wf-scroll {
  flex: 1;
  min-width: 0;
  overflow-x: auto;
  overflow-y: hidden;
  position: relative;
  background: var(--bg-card);
}
.wf-scroll::-webkit-scrollbar { height: 8px; }
.wf-events {
  display: flex;
  flex-direction: row;
  width: max-content;
  min-width: 100%;
  height: 100%;
  position: relative;
}

/* ─── Event column ──────────────────────────────────────────────────────── */
.ev-col {
  display: grid;
  grid-template-rows: var(--rows, 76px);
  width: 158px;
  flex-shrink: 0;
  position: relative;
  padding: 0;
  animation: col-enter 0.18s ease-out;
}
@keyframes col-enter {
  from { opacity: 0; transform: translateX(8px); }
  to   { opacity: 1; transform: translateX(0); }
}
.ev-col.msg-col svg.msg-arrow {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  overflow: visible;
  z-index: 1;
}
.ev-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  padding: 8px 10px;
}
/* Per-cell lifeline rail (only inside the events area) */
.ev-cell::before {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  top: 50%;
  height: 1px;
  background: var(--rail-color, transparent);
  pointer-events: none;
  z-index: 0;
}

/* ─── Event boxes (sherbet bars, navy ink) ─────────────────────────────── */
.ev-box {
  width: 100%;
  max-width: 138px;
  background: var(--bar-fill);
  color: var(--ink);
  border: 1px solid var(--bar-edge);
  border-radius: 6px;
  padding: 7px 11px;
  cursor: pointer;
  transition: background 0.12s ease, border-color 0.12s ease,
              box-shadow 0.16s ease, transform 0.08s ease;
  position: relative;
  z-index: 2;
  user-select: none;
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-height: 42px;
  --bar-fill: var(--k-action);
  --bar-edge: rgba(20,49,70,0.16);
}
.ev-box:hover {
  filter: brightness(1.04);
  box-shadow: 0 4px 12px rgba(20,49,70,0.10);
  transform: translateY(-1px);
}
.ev-box.selected {
  box-shadow: 0 0 0 2px var(--bg-card), 0 0 0 4px var(--bar-fill);
}
.ev-box-tag {
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink);
  opacity: 0.78;
  display: flex;
  align-items: center;
  gap: 6px;
}
.ev-box-name {
  font-size: 12.5px;
  font-weight: 600;
  color: var(--ink);
  letter-spacing: -0.01em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Type-specific bar fills */
.ev-box.act-box { --bar-fill: var(--k-action); }
.ev-box.act-box.planner-act {
  --bar-fill: var(--k-model);
}
.ev-box.msg-box.send  { --bar-fill: var(--k-send);  }
.ev-box.msg-box.recv  { --bar-fill: var(--k-recv);  }
.ev-box.msg-box.ctrl  { --bar-fill: var(--k-ctrl); border-style: dashed; opacity: 0.92; }
.ev-box.dec-box       { --bar-fill: var(--k-decision); }
.ev-box.dec-box .ev-box-name {
  display: flex;
  align-items: center;
  gap: 5px;
}
.dec-symbol {
  font-size: 13px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  color: var(--ink);
}

/* Action box pulse while running */
.ev-box.act-box.running {
  animation: act-pulse 1.4s ease-in-out infinite;
}
@keyframes act-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(20,49,70,0); }
  50%      { box-shadow: 0 0 0 4px rgba(105,166,224,0.20); }
}


/* ─── Detail panel (push, not overlay) ──────────────────────────────────── */
#detail-panel {
  width: 360px;
  flex-shrink: 0;
  background: var(--bg-card);
  border-left: 1px solid var(--hairline);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  --accent: var(--k-action);
}
#detail-panel[hidden] { display: none !important; }
.detail-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--hairline);
  background: var(--bg);
  flex-shrink: 0;
}
.detail-type {
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink);
  padding: 4px 9px;
  background: var(--accent);
  border-radius: 4px;
  flex-shrink: 0;
}
.detail-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--ink);
  letter-spacing: -0.01em;
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.detail-close {
  width: 28px;
  height: 28px;
  background: var(--bg-card);
  border: 1px solid var(--hairline);
  color: var(--ink-mute);
  font-size: 14px;
  cursor: pointer;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: color 0.12s, border-color 0.12s, background 0.12s;
  font-weight: 600;
}
.detail-close:hover {
  color: var(--ink);
  border-color: var(--ink);
  background: var(--bg);
}
.detail-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px 16px 32px;
}
.detail-meta {
  display: flex;
  align-items: center;
  padding: 10px 12px;
  margin-bottom: 18px;
  border: 1px solid var(--hairline);
  background: var(--bg);
  border-radius: 8px;
}
.detail-meta-row {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12.5px;
  color: var(--ink);
  font-weight: 600;
  letter-spacing: -0.005em;
}
.detail-meta-arrow {
  color: var(--ink-faint);
  font-size: 12px;
  margin: 0 2px;
}
.detail-section { margin-bottom: 18px; }
.detail-section-label {
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin-bottom: 7px;
  display: flex;
  align-items: center;
  gap: 7px;
}
.detail-section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--hairline);
}
.detail-kv {
  border: 1px solid var(--hairline);
  background: var(--bg);
  border-radius: 8px;
  overflow: hidden;
}
.detail-kv-row {
  display: grid;
  grid-template-columns: minmax(60px, max-content) 1fr;
  gap: 14px;
  padding: 8px 12px;
  align-items: baseline;
}
.detail-kv-row + .detail-kv-row {
  border-top: 1px solid var(--hairline);
}
.detail-kv-key {
  font-size: 11px;
  font-weight: 600;
  color: var(--ink-mute);
  letter-spacing: -0.005em;
}
.detail-kv-val {
  font-size: 11.5px;
  color: var(--ink);
  word-break: break-word;
  white-space: pre-wrap;
  line-height: 1.55;
  font-family: 'JetBrains Mono', monospace;
}
.detail-empty {
  font-size: 11px;
  color: var(--ink-faint);
  font-style: italic;
  letter-spacing: -0.005em;
  text-align: center;
  padding: 10px;
  border: 1px dashed var(--hairline);
  border-radius: 6px;
  background: var(--bg-card);
}

/* ─── Vertical message arrows ───────────────────────────────────────────── */
svg.msg-arrow line {
  stroke-width: 1.6;
  stroke-linecap: round;
}
svg.msg-arrow.ctrl line {
  stroke-dasharray: 4 3;
}

/* ─── Responsive ─────────────────────────────────────────────────────────── */
@media (max-width: 1080px) {
  #sidebar { width: 200px; }
  #detail-panel { width: 320px; }
}
@media (max-width: 820px) {
  #sidebar { width: 176px; }
  #detail-panel { width: 280px; }
  .ll-strip { width: 132px; }
  .brand { padding: 0 14px; }
}
</style>
</head>
<body>

<header id="topbar">
  <div class="brand">
    <div class="brand-mark"></div>
    <div class="brand-name">Zipper<span class="brand-name-accent">Chat</span></div>
  </div>
  <div id="topbar-meta">
    <div id="status-indicator">
      <span class="status-dot"></span>
      <span class="status-label">connecting</span>
    </div>
  </div>
  <button id="dark-toggle" type="button" title="Toggle dark mode" aria-label="Toggle dark mode">
    <svg class="icon-sun" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
    <svg class="icon-moon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
  </button>
</header>

<div id="body">

  <aside id="sidebar">
    <div class="sidebar-section">
      <h3 class="section-label">Workflows</h3>
      <ul id="wf-tree"></ul>
    </div>
    <div class="sidebar-bottom">
      <button id="replay-btn" type="button" disabled>
        <span class="btn-icon">▶</span>
        <span class="btn-label">Run again</span>
      </button>
    </div>
  </aside>

  <main id="main">
    <div id="diagram-root"></div>
  </main>

  <aside id="detail-panel" hidden aria-hidden="true">
    <header class="detail-header">
      <span class="detail-type" id="detail-type">EVENT</span>
      <span class="detail-name" id="detail-name"></span>
      <button class="detail-close" type="button" aria-label="Close detail" title="Close (Esc)">✕</button>
    </header>
    <div class="detail-body" id="detail-body"></div>
  </aside>

</div>

<script>
'use strict';

// ─── Configuration ────────────────────────────────────────────────────────
const AGENT_COLORS = ['#3F5A78', '#5A7390', '#6E8298', '#7E8B9C', '#8E96A2'];
const ROW_HEIGHT = 76;
const STRIP_WIDTH = 152;

// Bar-fill kinds
const KIND = {
  action:   '#69A6E0',
  model:    '#9C8FD9',
  send:     '#7FB1B0',
  recv:     '#BFD9D8',
  ctrl:     '#C4CCD4',
  decision: '#E06B8A',
};

// ─── State ────────────────────────────────────────────────────────────────
const levels = new Map();          // pathKey → Level
let selectedBox = null;
let eventSrc = null;
let focusedLevelKey = null;

// ─── DOM refs ─────────────────────────────────────────────────────────────
const diagramRoot   = document.getElementById('diagram-root');
const statusDot     = document.querySelector('#status-indicator .status-dot');
const statusLabel   = document.querySelector('#status-indicator .status-label');
const statusInd     = document.getElementById('status-indicator');
const wfTreeEl      = document.getElementById('wf-tree');
const replayBtn     = document.getElementById('replay-btn');
const detailPanel   = document.getElementById('detail-panel');
const detailType    = document.getElementById('detail-type');
const detailName    = document.getElementById('detail-name');
const detailBody    = document.getElementById('detail-body');
const detailClose   = document.querySelector('.detail-close');
const darkToggleBtn = document.getElementById('dark-toggle');

// ─── Dark-mode toggle ─────────────────────────────────────────────────────
(function() {
  const saved = localStorage.getItem('zc-theme');
  if (saved === 'dark') document.body.classList.add('dark');
})();
darkToggleBtn.addEventListener('click', () => {
  const isDark = document.body.classList.toggle('dark');
  localStorage.setItem('zc-theme', isDark ? 'dark' : 'light');
});

// Expose strip width to CSS
document.documentElement.style.setProperty('--strip-w', STRIP_WIDTH + 'px');

// ─── Helpers ──────────────────────────────────────────────────────────────
function pathKey(p) { return p.length ? p.join('\x00') : '__root'; }
function pad2(n) { return n < 10 ? '0' + n : '' + n; }
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
                  .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function agentColor(i) { return AGENT_COLORS[i % AGENT_COLORS.length]; }
function hexToRgba(hex, alpha) {
  const h = hex.replace('#','');
  const r = parseInt(h.slice(0,2), 16);
  const g = parseInt(h.slice(2,4), 16);
  const b = parseInt(h.slice(4,6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
function isCtrlVal(v) {
  return typeof v === 'string' && v.startsWith('κ_ctrl_');
}
function isCtrlVals(vs) {
  return Array.isArray(vs) && vs.some(isCtrlVal);
}
function fmtValue(v, maxLen) {
  if (v === null || v === undefined) return 'null';
  if (typeof v === 'string' && v.startsWith('κ_ctrl_')) {
    return v.slice('κ_ctrl_'.length) || 'ctrl';
  }
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'string') {
    if (maxLen && v.length > maxLen) return v.slice(0, maxLen - 1) + '…';
    return v;
  }
  if (Array.isArray(v)) {
    return v.map(x => fmtValue(x, maxLen)).join(', ');
  }
  if (typeof v === 'object') {
    try { return JSON.stringify(v, null, 2); } catch { return String(v); }
  }
  return String(v);
}

// ─── Status indicator ─────────────────────────────────────────────────────
function setStatus(state, text) {
  statusDot.className = 'status-dot ' + (state || '');
  statusInd.classList.remove('connected','running','done','error');
  if (state) statusInd.classList.add(state);
  statusLabel.textContent = text;
}

// ─── Workflow tree ────────────────────────────────────────────────────────
function renderTree() {
  wfTreeEl.innerHTML = '';
  const root = levels.get(pathKey([]));
  if (!root) return;
  wfTreeEl.appendChild(renderTreeNode(root));
}
function renderTreeNode(lev) {
  const li = document.createElement('li');
  li.className = 'wf-tree-item';
  const row = document.createElement('div');
  row.className = 'wf-tree-row ' + (lev.status || '');
  row.dataset.key = lev.key;
  if (lev.key === focusedLevelKey) row.classList.add('focused');
  row.innerHTML = `
    <span class="wf-tree-status ${lev.status || ''}"></span>
    <span class="wf-tree-name">${escHtml(lev.name)}</span>
  `;
  row.addEventListener('click', () => {
    if (lev._plannerBox) {
      // Nested level — selecting it is the same as selecting its planner box
      fillDetailPanel(lev._plannerBox, lev._plannerBox._dataBuilder());
      requestAnimationFrame(() => scrollToLevel(lev));
    } else {
      // Direct workflow selection: deselect any box, highlight this level
      if (selectedBox) { selectedBox.classList.remove('selected'); selectedBox = null; }
      detailPanel.hidden = true;
      detailPanel.setAttribute('aria-hidden', 'true');
      setGroupActive(lev.groupEl, hexToRgba(lev.accent, 0.38));
      focusedLevelKey = lev.key;
      renderTree();
      scrollToLevel(lev);
    }
  });
  li.appendChild(row);
  const childList = Array.from(lev.children.values());
  if (childList.length > 0) {
    const ul = document.createElement('ul');
    ul.className = 'wf-tree-children';
    childList.forEach(c => ul.appendChild(renderTreeNode(c)));
    li.appendChild(ul);
  }
  return li;
}
function scrollToLevel(lev) {
  if (!lev || !lev.groupEl) return;
  lev.groupEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Lifeline status (in diagram strip) ──────────────────────────────────
function setLLStatusInLevel(lev, lifelineName, state) {
  const idx = lev.lifelineNames.indexOf(lifelineName);
  if (idx < 0 || !lev.stripEl) return;
  const row = lev.stripEl.children[idx];
  if (!row) return;
  row.classList.remove('thinking', 'done');
  if (state) row.classList.add(state);
}
function markAllLLDone(lev) {
  if (!lev.stripEl) return;
  for (const row of lev.stripEl.children) {
    row.classList.remove('thinking');
    row.classList.add('done');
  }
}

// ─── Level management ─────────────────────────────────────────────────────
function makeLevel(path, name, lifelineNames, parentLevel = null,
                   parentLifeline = null, parentSeq = null) {
  const key = pathKey(path);
  if (levels.has(key)) return levels.get(key);

  let accent = KIND.model;
  if (parentLevel && parentLifeline) {
    const idx = parentLevel.lifelineNames.indexOf(parentLifeline);
    if (idx >= 0) accent = agentColor(idx);
  }

  const lev = {
    path, key, name, lifelineNames,
    N: lifelineNames.length,
    parentLevel, parentLifeline, parentSeq,
    accent,
    status: 'running',
    children: new Map(),
    groupEl: null, bodyEl: null, stripEl: null, scrollEl: null, eventsEl: null, childrenEl: null,
    cgRows: {},
    cgIdSeq: 0,
    cgLast: {},
    cgPending: {},
    cgOpen: {},
    completed: [],
    pendingActBoxes: {},
    seqToActBox: new Map(),
  };
  levels.set(key, lev);
  if (parentLevel) {
    parentLevel.children.set(key, lev);
    if (parentSeq != null) markCallerAsPlanner(parentLevel, parentSeq, lev);
  }
  buildLevelDom(lev);
  renderTree();
  return lev;
}

function markCallerAsPlanner(parentLev, parentSeq, childLevel) {
  const box = parentLev.seqToActBox.get(parentSeq);
  if (!box) return;
  box.classList.add('planner-act');
  box._childLevel = childLevel;
  childLevel._plannerBox = box;
}

function buildLevelDom(lev) {
  const isNested = lev.path.length > 0;

  const group = document.createElement('section');
  group.className = 'wf-group' + (isNested ? ' nested' : '');
  group.dataset.path = lev.key;
  group.style.setProperty('--wf-accent', lev.accent);
  group.style.setProperty('--spawner-color', hexToRgba(lev.accent, 0.38));

  // Label
  const label = document.createElement('header');
  label.className = 'wf-label';
  const ctxHtml = (isNested && lev.parentLifeline)
    ? `<span class="wf-label-context">spawned by <strong>${escHtml(lev.parentLifeline)}</strong></span>`
    : '';
  const llHtml = isNested
    ? `<span class="wf-label-lifelines">${lev.lifelineNames.map(escHtml).join(' · ')}</span>`
    : '';
  label.innerHTML = `
    <span class="wf-label-name">${escHtml(lev.name)}</span>
    ${llHtml}
    ${ctxHtml}
  `;
  // Clicking a group's header: nested with planner → show detail; otherwise → select workflow
  label.addEventListener('click', () => {
    if (lev._plannerBox) {
      fillDetailPanel(lev._plannerBox, lev._plannerBox._dataBuilder());
    } else {
      if (selectedBox) { selectedBox.classList.remove('selected'); selectedBox = null; }
      detailPanel.hidden = true;
      detailPanel.setAttribute('aria-hidden', 'true');
      setGroupActive(lev.groupEl, hexToRgba(lev.accent, 0.38));
      focusedLevelKey = lev.key;
      renderTree();
    }
  });
  group.appendChild(label);

  // Body
  const body = document.createElement('div');
  body.className = 'wf-body';

  // Lifeline strip (white, no rail through it)
  const strip = document.createElement('aside');
  strip.className = 'll-strip';
  lev.lifelineNames.forEach((name, i) => {
    const c = agentColor(i);
    const row = document.createElement('div');
    row.className = 'll-strip-row';
    row.style.setProperty('--ll-color', c);
    row.style.setProperty('--row-h', ROW_HEIGHT + 'px');
    row.innerHTML = `
      <span class="ll-strip-swatch"></span>
      <span class="ll-strip-name">${escHtml(name)}</span>
      <span class="ll-strip-status"></span>
    `;
    strip.appendChild(row);
  });

  // Scroll + events
  const scroll = document.createElement('div');
  scroll.className = 'wf-scroll';
  const events = document.createElement('div');
  events.className = 'wf-events';
  events.style.setProperty('--rows', `repeat(${lev.N}, ${ROW_HEIGHT}px)`);
  scroll.appendChild(events);

  body.appendChild(strip);
  body.appendChild(scroll);
  group.appendChild(body);

  // Children container — nested sub-workflow groups go here (continuous guide line)
  const childrenEl = document.createElement('div');
  childrenEl.className = 'wf-children';
  group.appendChild(childrenEl);

  lev.groupEl = group;
  lev.bodyEl  = body;
  lev.stripEl = strip;
  lev.scrollEl = scroll;
  lev.eventsEl = events;
  lev.childrenEl = childrenEl;

  // Insert into DOM: nested groups go into their parent's children container
  if (isNested && lev.parentLevel && lev.parentLevel.childrenEl) {
    lev.parentLevel.childrenEl.appendChild(group);
  } else {
    diagramRoot.appendChild(group);
  }
}

function setLevelStatus(lev, status) {
  if (lev.status === status) return;
  lev.status = status;
  if (status === 'done' && lev.stripEl) markAllLLDone(lev);
  renderTree();
}

// ─── Constraint graph (per-level) ─────────────────────────────────────────
function newCgRow(lev, kind) {
  const r = {
    id: lev.cgIdSeq++, kind, level: 0,
    succ: new Set(), pred: new Set(),
    wrapper: null, cells: null, svg: null,
    fromIdx: -1, toIdx: -1, ctrl: false, color: null,
    occupied: new Set()
  };
  lev.cgRows[r.id] = r;
  return r;
}
function cgAddEdge(lev, u, v) {
  if (!u || !v || u.id === v.id) return;
  if (u.succ.has(v.id)) return;
  u.succ.add(v.id); v.pred.add(u.id);
  if (v.level <= u.level) cgRaise(lev, v, u.level + 1);
}
function cgRaise(lev, v, nl) {
  if (nl <= v.level) return;
  v.level = nl;
  for (const wId of v.succ) cgRaise(lev, lev.cgRows[wId], v.level + 1);
}
function hasPath(lev, start, target) {
  if (!start || !target) return false;
  if (start.id === target.id) return true;
  const seen = new Set([start.id]);
  const stack = [...start.succ];
  while (stack.length) {
    const id = stack.pop();
    if (id === target.id) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    const row = lev.cgRows[id];
    if (!row) continue;
    for (const succ of row.succ) stack.push(succ);
  }
  return false;
}
function canReuseActRow(lev, lifeline, r) {
  if (!r || r.kind !== 'action') return false;
  const idx = lev.lifelineNames.indexOf(lifeline);
  if (idx < 0 || r.occupied.has(idx)) return false;
  const last = lev.cgLast[lifeline];
  if (last && hasPath(lev, r, last)) return false;
  for (const pending of (lev.cgPending[lifeline] || [])) {
    if (pending.id !== r.id && hasPath(lev, pending, r)) return false;
  }
  return true;
}
function findReusableActRow(lev, lifeline) {
  return Object.values(lev.cgRows)
    .filter(r => canReuseActRow(lev, lifeline, r))
    .sort((a, b) => b.level !== a.level ? b.level - a.level : b.id - a.id)[0] || null;
}
function materializeOnLifeline(lev, lifeline, r) {
  cgAddEdge(lev, lev.cgLast[lifeline], r);
  for (const p of (lev.cgPending[lifeline] || [])) {
    if (p.id !== r.id) cgAddEdge(lev, r, p);
  }
  lev.cgLast[lifeline] = r;
}
function syncOrder(lev) {
  const sorted = Object.values(lev.cgRows)
    .filter(r => r.wrapper)
    .sort((a, b) => a.level !== b.level ? a.level - b.level : a.id - b.id);
  for (const r of sorted) lev.eventsEl.appendChild(r.wrapper);
  requestAnimationFrame(() => redrawArrowsForLevel(lev));
}
function scrollLevelToEnd(lev) {
  if (lev.scrollEl) lev.scrollEl.scrollLeft = lev.scrollEl.scrollWidth;
}

// ─── Detail panel ─────────────────────────────────────────────────────────
function fillDetailPanel(box, data) {
  if (selectedBox && selectedBox !== box) {
    selectedBox.classList.remove('selected');
  }
  selectedBox = box;
  box.classList.add('selected');

  // Active group = sub-workflow (if this box spawns one), otherwise this box's own level
  const activeLev = box._childLevel || box._level;
  if (activeLev && activeLev.groupEl) {
    setGroupActive(activeLev.groupEl, hexToRgba(activeLev.accent, 0.38));
    focusedLevelKey = activeLev.key;
    renderTree();
  }

  const accent = data.accent || KIND.action;
  detailPanel.style.setProperty('--accent', accent);
  detailType.textContent = data.type;
  detailName.textContent = data.name || '';
  detailBody.innerHTML = '';

  // Meta block
  const meta = document.createElement('div');
  meta.className = 'detail-meta';
  let metaHtml;
  if (data.lifelines) {
    metaHtml = `<span class="detail-meta-row">
      <span>${escHtml(data.lifelines[0])}</span>
      <span class="detail-meta-arrow">→</span>
      <span>${escHtml(data.lifelines[1])}</span>
    </span>`;
  } else if (data.lifeline) {
    metaHtml = `<span class="detail-meta-row"><span>${escHtml(data.lifeline)}</span></span>`;
  } else {
    metaHtml = `<span class="detail-meta-row"><span>—</span></span>`;
  }
  meta.innerHTML = metaHtml;
  detailBody.appendChild(meta);

  // Sections
  if (data.sections) {
    data.sections.forEach(sec => {
      const section = document.createElement('div');
      section.className = 'detail-section';
      const lbl = document.createElement('div');
      lbl.className = 'detail-section-label';
      lbl.textContent = sec.label;
      section.appendChild(lbl);

      const entries = sec.entries || {};
      const keys = Object.keys(entries).filter(k => {
        const v = entries[k];
        if (Array.isArray(v) && v.every(isCtrlVal)) return false;
        return true;
      });

      if (keys.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'detail-empty';
        empty.textContent = sec.emptyText || '(none)';
        section.appendChild(empty);
      } else {
        const kv = document.createElement('div');
        kv.className = 'detail-kv';
        keys.forEach(k => {
          const row = document.createElement('div');
          row.className = 'detail-kv-row';
          row.innerHTML = `
            <span class="detail-kv-key">${escHtml(k)}</span>
            <span class="detail-kv-val">${escHtml(fmtValue(entries[k]))}</span>
          `;
          kv.appendChild(row);
        });
        section.appendChild(kv);
      }
      detailBody.appendChild(section);
    });
  }

  detailPanel.hidden = false;
  detailPanel.setAttribute('aria-hidden', 'false');
  requestAnimationFrame(redrawAllArrows);
}
let activeGroupEl = null;
function setGroupActive(groupEl, color) {
  if (activeGroupEl) {
    activeGroupEl.classList.remove('wf-active');
    activeGroupEl.style.removeProperty('--spawner-color');
  }
  activeGroupEl = groupEl || null;
  if (activeGroupEl) {
    activeGroupEl.style.setProperty('--spawner-color', color || 'rgba(156,143,217,0.4)');
    activeGroupEl.classList.add('wf-active');
  }
}
function closeDetail() {
  if (selectedBox) {
    selectedBox.classList.remove('selected');
    selectedBox = null;
  }
  setGroupActive(null);
  focusedLevelKey = null;
  renderTree();
  detailPanel.hidden = true;
  detailPanel.setAttribute('aria-hidden', 'true');
  requestAnimationFrame(redrawAllArrows);
}
detailClose.addEventListener('click', closeDetail);
diagramRoot.addEventListener('click', (e) => {
  if (e.target.closest('.ev-box')) return;
  if (e.target.closest('.wf-label')) return;
  closeDetail();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !detailPanel.hidden) closeDetail();
});

// ─── Box factory ──────────────────────────────────────────────────────────
function makeBox(opts) {
  const box = document.createElement('div');
  box.className = `ev-box ${opts.kind || ''}`;
  if (opts.extraClass) box.classList.add(opts.extraClass);
  box.innerHTML = `
    <div class="ev-box-tag">${opts.tag}</div>
    <div class="ev-box-name">${escHtml(opts.name)}</div>
  `;
  box._dataBuilder = opts.dataBuilder;
  box.addEventListener('click', (e) => {
    e.stopPropagation();
    if (selectedBox === box && !detailPanel.hidden) {
      closeDetail();
      return;
    }
    fillDetailPanel(box, opts.dataBuilder());
  });
  return box;
}

// ─── Action rendering ─────────────────────────────────────────────────────
function handleActStart(lev, ev) {
  const i = lev.lifelineNames.indexOf(ev.lifeline);
  if (i < 0) return;

  setLLStatusInLevel(lev, ev.lifeline, 'thinking');

  let r = findReusableActRow(lev, ev.lifeline);
  if (r) {
    materializeOnLifeline(lev, ev.lifeline, r);
  } else {
    r = newCgRow(lev, 'action');
    const wrapper = document.createElement('div');
    wrapper.className = 'ev-col act-col';
    wrapper.style.setProperty('--rows', `repeat(${lev.N}, ${ROW_HEIGHT}px)`);
    r.wrapper = wrapper;
    r.cells = lev.lifelineNames.map((_, k) => {
      const cell = document.createElement('div');
      cell.className = 'ev-cell';
      cell.dataset.row = String(k);
      cell.style.setProperty('--rail-color', hexToRgba(agentColor(k), 0.55));
      wrapper.appendChild(cell);
      return cell;
    });
    materializeOnLifeline(lev, ev.lifeline, r);
    lev.eventsEl.appendChild(wrapper);
  }
  r.occupied.add(i);
  r.seq = ev.seq;

  const inputs = ev.inputs || {};
  const box = makeBox({
    kind: 'act-box',
    extraClass: 'running',
    tag: 'ACT',
    name: ev.action,
    dataBuilder: () => ({
      type: 'ACT',
      name: ev.action,
      lifeline: ev.lifeline,
      seq: ev.seq,
      accent: box.classList.contains('planner-act') ? KIND.model : KIND.action,
      sections: [
        { label: 'Inputs',  entries: inputs,           emptyText: '(no inputs)' },
        { label: 'Outputs', entries: box._outputs || {}, emptyText: '(awaiting outputs…)' },
      ],
    }),
  });
  box._outputs = {};
  box._level = lev;
  r.cells[i].appendChild(box);
  lev.pendingActBoxes[`${ev.lifeline}:${ev.seq}`] = box;
  lev.seqToActBox.set(ev.seq, box);
  syncOrder(lev);
  scrollLevelToEnd(lev);
}

function handleAct(lev, ev) {
  const key = `${ev.lifeline}:${ev.seq}`;
  const box = lev.pendingActBoxes[key];
  if (!box) return;
  delete lev.pendingActBoxes[key];
  box.classList.remove('running');
  box._outputs = ev.outputs || {};
  setLLStatusInLevel(lev, ev.lifeline, '');

  if (selectedBox === box) {
    const accent = box.classList.contains('planner-act') ? KIND.model : KIND.action;
    fillDetailPanel(box, {
      type: box.classList.contains('planner-act') ? 'PLANNER ACT' : 'ACT',
      name: ev.action,
      lifeline: ev.lifeline,
      seq: ev.seq,
      accent,
      sections: [
        { label: 'Inputs',  entries: ev.inputs || {},  emptyText: '(no inputs)' },
        { label: 'Outputs', entries: ev.outputs || {}, emptyText: '(no outputs)' },
      ],
    });
  }
}

// ─── Decision rendering ───────────────────────────────────────────────────
function handleDecision(lev, ev) {
  const i = lev.lifelineNames.indexOf(ev.lifeline);
  if (i < 0) return;

  const r = newCgRow(lev, 'decision');
  const wrapper = document.createElement('div');
  wrapper.className = 'ev-col dec-col';
  wrapper.style.setProperty('--rows', `repeat(${lev.N}, ${ROW_HEIGHT}px)`);
  r.wrapper = wrapper;
  r.cells = lev.lifelineNames.map((_, k) => {
    const cell = document.createElement('div');
    cell.className = 'ev-cell';
    cell.style.setProperty('--rail-color', hexToRgba(agentColor(k), 0.55));
    wrapper.appendChild(cell);
    return cell;
  });
  materializeOnLifeline(lev, ev.lifeline, r);

  const isTrue = !!ev.value;
  const isWhile = ev.kind === 'while';
  let sym, word;
  if (isWhile) {
    sym = isTrue ? '↻' : '⊥';
    word = isTrue ? 'continue' : 'exit';
  } else {
    sym = isTrue ? '⊤' : '⊥';
    word = isTrue ? 'true' : 'false';
  }
  const tag = isWhile ? 'WHILE' : 'IF';

  const box = document.createElement('div');
  box.className = 'ev-box dec-box';
  box._level = lev;
  box.innerHTML = `
    <div class="ev-box-tag">${tag}</div>
    <div class="ev-box-name">
      <span class="dec-symbol">${sym}</span>
      <span>${escHtml(word)}</span>
    </div>
  `;
  box.addEventListener('click', (e) => {
    e.stopPropagation();
    if (selectedBox === box && !detailPanel.hidden) {
      closeDetail();
      return;
    }
    fillDetailPanel(box, {
      type: tag,
      name: word,
      lifeline: ev.lifeline,
      accent: KIND.decision,
      sections: [
        { label: 'Condition', entries: ev.condition ? { expr: ev.condition } : {},
          emptyText: '(no condition expression)' },
        { label: 'Verdict', entries: { value: isTrue } },
      ],
    });
  });
  r.cells[i].appendChild(box);
  syncOrder(lev);
  scrollLevelToEnd(lev);
}

// ─── Message rendering ────────────────────────────────────────────────────
function handleSend(lev, ev) {
  const fromIdx = lev.lifelineNames.indexOf(ev.from);
  const toIdx   = lev.lifelineNames.indexOf(ev.to);
  if (fromIdx < 0 || toIdx < 0) return;
  const ctrl = isCtrlVals(ev.values);

  const r = newCgRow(lev, 'message');
  r.fromIdx = fromIdx; r.toIdx = toIdx; r.ctrl = ctrl;
  r.color = ctrl ? KIND.ctrl : KIND.send;

  const wrapper = document.createElement('div');
  wrapper.className = 'ev-col msg-col';
  wrapper.style.setProperty('--rows', `repeat(${lev.N}, ${ROW_HEIGHT}px)`);
  r.wrapper = wrapper;
  r.cells = lev.lifelineNames.map((_, k) => {
    const cell = document.createElement('div');
    cell.className = 'ev-cell';
    cell.style.setProperty('--rail-color', hexToRgba(agentColor(k), 0.55));
    wrapper.appendChild(cell);
    return cell;
  });

  const sendBindings = ctrl ? { branch: ev.values[0] } : (ev.bindings || {});
  const sendBox = makeBox({
    kind: 'msg-box send',
    extraClass: ctrl ? 'ctrl' : '',
    tag: ctrl ? 'CTRL' : 'SEND',
    name: '→ ' + ev.to,
    dataBuilder: () => ({
      type: ctrl ? 'CTRL SEND' : 'SEND',
      name: `${ev.from} → ${ev.to}`,
      lifelines: [ev.from, ev.to],
      fromColor: agentColor(fromIdx),
      toColor:   agentColor(toIdx),
      seq: ev.seq,
      accent: ctrl ? KIND.ctrl : KIND.send,
      sections: [
        { label: 'Values', entries: sendBindings, emptyText: '(no values)' },
      ],
    }),
  });
  sendBox._level = lev;
  r.cells[fromIdx].appendChild(sendBox);

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', ctrl ? 'msg-arrow ctrl' : 'msg-arrow');
  wrapper.appendChild(svg);
  r.svg = svg;

  materializeOnLifeline(lev, ev.from, r);

  const chanKey = `${ev.from}->${ev.to}`;
  if (!lev.cgOpen[chanKey]) lev.cgOpen[chanKey] = [];
  lev.cgOpen[chanKey].push(r);
  if (!lev.cgPending[ev.to]) lev.cgPending[ev.to] = new Set();
  lev.cgPending[ev.to].add(r);

  lev.completed.push(r);
  lev.eventsEl.appendChild(wrapper);
  syncOrder(lev);
  scrollLevelToEnd(lev);
}

function handleRecv(lev, ev) {
  const chanKey = `${ev.from}->${ev.to}`;
  const queue   = lev.cgOpen[chanKey];
  if (!queue || !queue.length) return;
  const r = queue.shift();
  lev.cgPending[ev.to]?.delete(r);

  const ctrl = r.ctrl;
  const toIdx = r.toIdx;
  const recvBindings = ev.bindings || {};

  const recvBox = makeBox({
    kind: 'msg-box recv',
    extraClass: ctrl ? 'ctrl' : '',
    tag: ctrl ? 'CTRL' : 'RECV',
    name: '← ' + ev.from,
    dataBuilder: () => ({
      type: ctrl ? 'CTRL RECV' : 'RECV',
      name: `${ev.from} → ${ev.to}`,
      lifelines: [ev.from, ev.to],
      fromColor: agentColor(r.fromIdx),
      toColor:   agentColor(toIdx),
      seq: ev.seq,
      accent: ctrl ? KIND.ctrl : KIND.recv,
      sections: [
        { label: 'Bindings', entries: recvBindings, emptyText: '(no bindings)' },
      ],
    }),
  });
  recvBox._level = lev;
  r.cells[toIdx].appendChild(recvBox);
  materializeOnLifeline(lev, ev.to, r);
  syncOrder(lev);
  scrollLevelToEnd(lev);
}

// ─── Vertical arrow drawing ───────────────────────────────────────────────
function drawArrow(lev, r) {
  const wrap = r.wrapper;
  const svg  = r.svg;
  if (!wrap || !svg) return;
  const W = wrap.clientWidth;
  const H = wrap.clientHeight || (lev.N * ROW_HEIGHT);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = '';

  const sendBox = r.cells[r.fromIdx].querySelector('.ev-box');
  const recvBox = r.cells[r.toIdx].querySelector('.ev-box');
  const wrapRect = wrap.getBoundingClientRect();
  const zoom = W > 0 ? wrapRect.width / W : 1;

  const x = W / 2;
  const dir = r.toIdx > r.fromIdx ? 1 : -1;
  const cellH = H / lev.N;

  let y1, y2;
  if (sendBox) {
    const rb = sendBox.getBoundingClientRect();
    y1 = (dir > 0 ? rb.bottom - wrapRect.top : rb.top - wrapRect.top) / zoom;
  } else {
    y1 = (r.fromIdx + 0.5) * cellH;
  }
  if (recvBox) {
    const rb = recvBox.getBoundingClientRect();
    y2 = (dir > 0 ? rb.top - wrapRect.top : rb.bottom - wrapRect.top) / zoom;
  } else {
    y2 = (r.toIdx + 0.5) * cellH;
  }

  const color = r.color;
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', x); line.setAttribute('y1', y1);
  line.setAttribute('x2', x); line.setAttribute('y2', y2);
  line.setAttribute('stroke', color);
  if (r.ctrl) line.setAttribute('stroke-dasharray', '4 3');
  svg.appendChild(line);

  const arrowLen = 9;
  const arrowHalf = 5;
  const ay = y2 - dir * arrowLen;
  const pts = `${x},${y2} ${x - arrowHalf},${ay} ${x + arrowHalf},${ay}`;
  const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
  arrow.setAttribute('points', pts);
  arrow.setAttribute('fill', color);
  svg.appendChild(arrow);
}
function redrawArrowsForLevel(lev) {
  for (const r of lev.completed) drawArrow(lev, r);
}
function redrawAllArrows() {
  for (const lev of levels.values()) redrawArrowsForLevel(lev);
}
new ResizeObserver(redrawAllArrows).observe(diagramRoot);
window.addEventListener('resize', redrawAllArrows);

// ─── Event dispatch ───────────────────────────────────────────────────────
function dispatchEvent(e) {
  if (e.type === 'init') { handleInit(e); return; }
  if (e.type === 'level_push') { handleLevelPush(e); return; }
  if (e.type === 'level_pop') {
    const lev = levels.get(pathKey(e.path || []));
    if (lev) setLevelStatus(lev, 'done');
    return;
  }
  if (e.type === 'done') {
    setStatus('done', 'done');
    replayBtn.disabled = false;
    const root = levels.get(pathKey([]));
    if (root) setLevelStatus(root, 'done');
    return;
  }
  if (e.type === 'close') {
    if (eventSrc) { eventSrc.close(); eventSrc = null; }
    setStatus('error', 'stopped');
    return;
  }

  const path = e.path || [];
  const lev = levels.get(pathKey(path));
  if (!lev) return;

  if (e.type === 'act_start') handleActStart(lev, e);
  else if (e.type === 'act')    handleAct(lev, e);
  else if (e.type === 'send')   handleSend(lev, e);
  else if (e.type === 'recv')   handleRecv(lev, e);
  else if (e.type === 'decision') handleDecision(lev, e);
}

function handleInit(e) {
  levels.clear();
  diagramRoot.innerHTML = '';
  closeDetail();
  replayBtn.disabled = true;
  focusedLevelKey = null;

  makeLevel([], 'workflow', e.lifelines);
  setStatus('running', 'running');
}

function handleLevelPush(e) {
  const childPath  = e.path;
  const parentPath = childPath.slice(0, -1);
  const parent = levels.get(pathKey(parentPath));
  if (!parent) return;
  let parentLifeline = null;
  for (const key of Object.keys(parent.pendingActBoxes)) {
    const [llname, seqStr] = key.split(':');
    if (parseInt(seqStr, 10) === e.parent_seq) {
      parentLifeline = llname;
      break;
    }
  }
  if (!parentLifeline) {
    const box = parent.seqToActBox.get(e.parent_seq);
    if (box) {
      const cell = box.parentElement;
      const rowIdx = cell ? parseInt(cell.dataset.row || '-1', 10) : -1;
      if (rowIdx >= 0) parentLifeline = parent.lifelineNames[rowIdx];
    }
  }
  makeLevel(childPath, e.name, e.lifelines, parent, parentLifeline, e.parent_seq);
}

// ─── Replay button ────────────────────────────────────────────────────────
replayBtn.addEventListener('click', () => {
  if (replayBtn.disabled) return;
  replayBtn.disabled = true;
  setStatus('connected', 'waiting…');
  fetch('/replay', { method: 'POST' }).then(() => connect());
});

// ─── SSE ──────────────────────────────────────────────────────────────────
function connect() {
  if (eventSrc) { eventSrc.close(); eventSrc = null; }
  const src = new EventSource('/events');
  eventSrc = src;
  src.onopen = () => { setStatus('connected', 'connected'); };
  src.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      dispatchEvent(data);
    } catch (err) {
      console.error('event parse error', err, ev.data);
    }
  };
  src.onerror = () => { setStatus('error', 'reconnecting…'); };
}

// Initial empty state
diagramRoot.innerHTML = '<div class="empty-msg">Awaiting workflow…</div>';
connect();
</script>

</body>
</html>
"""
