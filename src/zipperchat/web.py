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
                  pending_human_inputs: dict,
                  pending_lock: threading.Lock):
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
                with pending_lock:
                    entry = pending_human_inputs.get(req_id) if req_id else None
                if entry is not None:
                    evt, result_box = entry
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
        self._pending_lock = threading.Lock()

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
            self._pending_human_inputs, self._pending_lock,
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
        lock = self._pending_lock

        def backend(action, inputs: dict) -> dict:
            req_id = str(uuid.uuid4())
            evt = threading.Event()
            result_box: list = []
            with lock:
                pending[req_id] = (evt, result_box)

            context_val = action.context.format(**inputs) if action.context else None
            instruction_val = action.instruction.format(**inputs) if action.instruction else None
            prefill_val = action.prefill.format(**inputs) if action.prefill else None

            lifeline_name = threading.current_thread().name
            request_event: dict[str, object] = {
                "type": "human_input_required",
                "id": req_id,
                "lifeline": lifeline_name,
                "kind": action.kind,
                "context": context_val,
                "instruction": instruction_val,
                "prefill": prefill_val,
                "submit_label": action.submit_label,
                "cancel_label": action.cancel_label,
            }
            if path is not None:
                request_event["path"] = list(path)
            self._bus.publish(request_event)

            evt.wait()
            with lock:
                pending.pop(req_id, None)
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
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
:root {
  --bg:              #F5F2EC;
  --panel:           #ffffff;
  --rule:            rgba(20,20,40,0.08);
  --text:            #14141A;
  --text-soft:       #3a3a5a;
  --text-mute:       #6e6e92;
  --text-faint:      #a8a8c4;
  --accent:          #2148FF;
  --accent-bg:       #e8edff;
  --accent-attn:     #E94F2E;
  --accent-attn-bg:  #fde8e3;
  --btn-bg:          #14141A;
  --btn-text:        #F5F2EC;
  --done-clr:        #9aaa2a;
  --serif:      'Instrument Serif', Georgia, serif;
  --sans:       'Space Grotesk', system-ui, sans-serif;
  --mono:       'Space Mono', 'Courier New', monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body { font-family: var(--sans); background: var(--bg); color: var(--text); font-size: 14px; }


/* ── App shell ──────────────────────────────────────────────────────────── */
#app  { display: flex; flex-direction: column; height: 100vh; }
#hdr  {
  display: flex; align-items: center; gap: 20px;
  padding: 0 56px 0 25px; height: 70px; flex-shrink: 0;
  border-bottom: 1px solid var(--rule); background: var(--bg);
}
#body {
  display: grid; grid-template-columns: 250px 1fr;
  flex: 1; min-height: 0; overflow: hidden;
}

/* ── Header ─────────────────────────────────────────────────────────────── */
.hdr-logo { height: 28px; display: block; }

.hdr-gap  { flex: 1; }
#your-turn {
  display: none; align-items: center; gap: 8px;
  font-size: 13px; color: var(--accent-attn); font-weight: 500;
}
#your-turn.on { display: flex; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
.yt-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--accent-attn); flex-shrink: 0;
  animation: pulse 1.2s ease-in-out infinite;
}
#yt-count {
  font-family: var(--mono); font-size: 11px;
  background: var(--accent-attn-bg); color: var(--accent-attn); border-radius: 10px; padding: 1px 7px;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
#sidebar { overflow-y: auto; border-right: 1px solid var(--rule); padding: 16px 0 32px; }
.sg-group { margin-bottom: 24px; }
.sg-group-inbox {
  border-bottom: 1px solid var(--rule);
  padding-bottom: 16px; margin-bottom: 16px;
}
.sg-hdr {
  display: flex; align-items: baseline; justify-content: space-between;
  padding: 0 28px 8px; font-size: 14px; font-weight: 500; color: var(--text);
  cursor: pointer; user-select: none;
}
.sg-hdr:hover { color: var(--text); }
.sg-chevron {
  display: inline-block; width: 8px; height: 8px; flex-shrink: 0;
  border-right: 1.5px solid var(--text-mute); border-bottom: 1.5px solid var(--text-mute);
  transform: rotate(45deg); transition: transform .15s;
  margin-right: 8px; position: relative; top: -1px;
}
.sg-group.sg-folded .sg-chevron { transform: rotate(-45deg); }
.sg-group.sg-folded .sg-row { display: none; }
.sg-hdr-inbox { font-weight: 600; }
.sg-count { font-family: var(--mono); font-size: 11px; color: var(--text-faint); }
.sg-row {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 28px; cursor: pointer;
  border-left: 2px solid transparent;
  font-size: 13px; color: var(--text-soft);
  user-select: none; transition: background .1s;
}
.sg-row:hover { background: rgba(0,0,0,0.03); }
.sg-row:focus { outline: none; background: rgba(0,0,0,0.05); }
.sg-row.sg-sel { background: var(--accent-bg); border-left-color: var(--accent); color: var(--text); }
.sg-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.dot-done    { background: transparent; border: 1.5px solid var(--text-faint); }
.dot-running { background: var(--text-faint); }
.dot-pending { background: var(--accent-attn); animation: pulse 1.2s ease-in-out infinite; }
.sg-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sg-you  { font-size: 11px; font-style: italic; color: var(--accent-attn); flex-shrink: 0; }
.sb-empty { padding: 40px 28px; font-size: 13px; color: var(--text-faint); }

/* ── Inspector ─────────────────────────────────────────────────────────────── */
#inspector { overflow-y: auto; }
#ins-body  { padding: 32px 56px; min-height: 100%; }
.ins-empty-state {
  display: flex; align-items: center; justify-content: center;
  min-height: 300px; font-size: 14px; color: var(--text-faint);
}

/* Meta */
.ins-meta {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  margin-bottom: 12px; font-size: 14px; color: var(--text-mute);
}
.ins-meta-ll   { font-weight: 500; color: var(--text-soft); }
.ins-meta-dot  { color: var(--text-faint); }
.ins-meta-kind { font-family: var(--mono); font-size: 12px; }
.ins-meta-time { font-family: var(--mono); font-size: 12px; color: var(--text-faint); }
.ins-meta-fn   { font-family: var(--mono); font-size: 12px; color: var(--text-faint); }
.ins-meta-await { margin-left: auto; font-size: 13px; color: var(--accent-attn); font-weight: 500; font-style: italic; }

/* Title */
.ins-title {
  font-family: var(--sans); font-size: 24px; font-weight: 600;
  letter-spacing: -0.01em; line-height: 1.2; color: var(--text); margin-bottom: 24px;
}

/* Sections */
.ins-section { margin-bottom: 24px; }
.ins-sec-label {
  font-size: 11px; font-weight: 600; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--text-mute); margin-bottom: 10px;
}
.ins-instr-label {
  font-size: 14px; font-style: italic; color: var(--text-mute); margin-bottom: 10px;
}
.ins-sec-body { font-size: 15px; line-height: 1.6; color: var(--text); }

/* Prompt context */
.ins-ctx {
  white-space: pre-wrap; word-break: break-word; color: var(--text-soft);
  background: rgba(20,18,12,0.035); border-radius: 6px; padding: 12px 14px;
  font-size: 15px; line-height: 1.6;
}
.ctx-hdr       { font-weight: 600; color: var(--text); }
.ctx-hint      { font-style: italic; color: var(--text-mute); }
.ins-ctx-mono  { font-family: var(--mono); font-size: 13px; }

/* Email artifacts — read (filled) and write (outlined) */
.ea { border-radius: 6px; padding: 20px 18px; }
.ea-read { background: rgba(20,18,12,0.035); white-space: pre-wrap; word-break: break-word; }
.ea-write { border: 1px solid rgba(20,20,40,0.13); transition: border-color .15s; }
.ea-write:focus-within { border-color: var(--accent); }
.ea-subj {
  font-family: var(--sans); font-size: 18px; font-weight: 500;
  color: var(--text); line-height: 1.3; margin-bottom: 4px; white-space: normal;
}
.ea-hdr   { font-size: 13px; color: var(--text-mute); }
.ea-instr { font-size: 13px; color: var(--text-mute); font-style: italic; }
.ea-rule  { border: none; border-top: 1px solid rgba(20,18,12,0.1); margin: 10px 0 12px; }
.ea-body  { font-size: 14px; line-height: 1.6; color: var(--text-soft); }
.ea-ta {
  width: 100%; font-family: var(--sans); font-size: 15px; line-height: 1.6;
  padding: 0; border: none; background: transparent; color: var(--text);
  resize: none; min-height: 160px; field-sizing: content; outline: none;
}
.ea-ta::placeholder { color: var(--text-faint); font-style: italic; }

/* Resolved */
.ins-resolved-val { color: var(--done-clr); font-style: italic; }

/* Textarea */
.ins-split {
  display: grid; grid-template-columns: 1fr 1.1fr; gap: 28px; align-items: start;
  margin-bottom: 24px;
}
.ins-split-work { display: flex; flex-direction: column; gap: 14px; }

.ins-ta {
  width: 100%; font-family: var(--sans); font-size: 15px; line-height: 1.6;
  padding: 20px 18px; border: 1px solid rgba(20,20,40,0.13); border-radius: 6px;
  background: var(--bg); color: var(--text); resize: none; min-height: 160px;
  field-sizing: content; outline: none; transition: border-color .15s;
}
.ins-ta::placeholder { color: var(--text-faint); font-style: italic; }
.ins-ta:focus { border-color: var(--accent); }

/* Action row */
.ins-actions { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 20px; }
.btn-approve {
  font-family: var(--sans); font-size: 14px; font-weight: 500;
  padding: 10px 28px; border-radius: 6px;
  background: var(--btn-bg); color: var(--btn-text);
  border: none; cursor: pointer; transition: opacity .15s;
}
.btn-approve:hover:not(:disabled) { opacity: .85; }
.btn-approve:disabled { opacity: .35; cursor: default; }
.btn-decline {
  font-family: var(--sans); font-size: 14px; color: var(--text-mute);
  background: none; border: none; cursor: pointer;
  text-decoration: underline; text-underline-offset: 3px; padding: 10px 4px;
}
.btn-decline:hover:not(:disabled) { color: var(--text); }
.btn-decline:disabled { opacity: .3; cursor: default; }
.btn-secondary {
  font-family: var(--sans); font-size: 14px; font-weight: 500;
  padding: 10px 28px; border-radius: 6px;
  background: transparent; color: var(--text-soft);
  border: 1px solid rgba(20,18,12,0.15); cursor: pointer; transition: all .15s;
}
.btn-secondary:hover:not(:disabled) { border-color: rgba(20,18,12,0.3); color: var(--text); }
.btn-secondary:disabled { opacity: .35; cursor: default; }
.ins-hint { margin-left: auto; font-size: 12px; color: var(--text-faint); font-style: italic; }

/* Choice buttons */
.ins-choices { gap: 8px; flex-wrap: wrap; margin-top: 16px; }
.btn-choice {
  font-family: var(--sans); font-size: 14px; padding: 9px 16px; border-radius: 6px;
  border: 1px solid var(--rule); background: var(--panel); color: var(--text-soft);
  cursor: pointer; transition: all .12s;
}
.btn-choice:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); background: var(--accent-bg); }
.btn-choice:disabled { opacity: .3; cursor: default; }

/* KvBlock */
.kv-row { display: flex; gap: 14px; padding: 6px 0; border-bottom: 1px solid var(--rule); }
.kv-row:last-child { border-bottom: none; }
.kv-key { font-family: var(--mono); font-size: 12px; color: var(--text-mute); min-width: 100px; flex-shrink: 0; padding-top: 2px; }
.kv-val { font-size: 15px; color: var(--text); line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
.kv-val-true  { font-family: var(--mono); color: var(--done-clr); }
.kv-val-false { font-family: var(--mono); color: var(--text-faint); }
.kv-empty     { font-size: 13px; color: var(--text-faint); font-style: italic; }

/* Scrollbars */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--rule); border-radius: 3px; }
</style>
</head>
<body>
<div id="app">
  <header id="hdr">
    <img src="/assets/zippergen-lockup-ink.svg" alt="ZipperGen" class="hdr-logo">
    <div class="hdr-gap"></div>
    <div id="your-turn">
      <span class="yt-dot"></span>
      Your turn
      <span id="yt-count">1</span>
    </div>
  </header>
  <div id="body">
    <div id="sidebar">
      <p class="sb-empty">Awaiting workflow&hellip;</p>
    </div>
    <div id="inspector">
      <div id="ins-body">
        <div class="ins-empty-state">Click an action to inspect it</div>
      </div>
    </div>
  </div>
</div>
<script>
// State
let evSrc         = null;
const byKey       = {};
const groups      = {};
const lifelines   = [];
const rowEls      = {};
const inboxRowEls = {};
const reqMap      = new Map();
let pending       = 0;
let inboxTotal    = 0;
let selectedId    = null;

// DOM
const sidebar  = document.getElementById('sidebar');
const insBody  = document.getElementById('ins-body');
const yourTurn = document.getElementById('your-turn');
const ytCount  = document.getElementById('yt-count');

// Helpers
function esc(s){ return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function isCtrl(v){ return typeof v==='string'&&v.startsWith('κ_ctrl_'); }
function fmtV(v){
  if(v===null||v===undefined) return '';
  if(isCtrl(v)) return '';
  if(typeof v==='boolean') return v?'true':'false';
  if(typeof v==='string') return v.length>600?v.slice(0,599)+'…':v;
  try{ return JSON.stringify(v,null,2); }catch{ return String(v); }
}
function refreshCount(){
  if(pending>0){ yourTurn.classList.add('on'); ytCount.textContent=pending; }
  else yourTurn.classList.remove('on');
  updateInboxBadge();
}

// Fold toggle
function toggleGroup(grpEl){ grpEl.classList.toggle('sg-folded'); }

// Inbox
function ensureInbox(){
  if(document.getElementById('grp-inbox')) return;
  const em=sidebar.querySelector('.sb-empty'); if(em) em.remove();
  const grp=document.createElement('div');
  grp.className='sg-group sg-group-inbox'; grp.id='grp-inbox';
  grp.innerHTML='<div class="sg-hdr sg-hdr-inbox"><span><span class="sg-chevron"></span>Inbox</span><span class="sg-count" id="grpc-inbox">0 / 0</span></div>';
  grp.querySelector('.sg-hdr').addEventListener('click', function(){ toggleGroup(grp); });
  sidebar.insertBefore(grp, sidebar.firstChild);
}

function inboxRowInner(key){
  const a=byKey[key];
  const req=a.reqId?reqMap.get(a.reqId):null;
  const hp=req&&!req.resolved&&a.kind==='human';
  const dc=hp?'dot-pending':'dot-done';
  return '<span class="sg-dot '+dc+'"></span><span class="sg-name">'+esc(a.name)+'</span>'+(hp?'<span class="sg-you">you</span>':'');
}

function createInboxRow(key){
  ensureInbox();
  inboxTotal++;
  const el=document.createElement('div');
  el.className='sg-row'; el.id='inbox-row-'+key; el.tabIndex=0;
  el.innerHTML=inboxRowInner(key);
  el.onclick=()=>selectAction(key);
  el.onkeydown=e=>{ if(e.key==='Enter'||e.key===' '){ e.preventDefault(); selectAction(key); } };
  inboxRowEls[key]=el;
  const grp=document.getElementById('grp-inbox');
  if(grp) grp.insertBefore(el, grp.querySelector('.sg-row'));
  updateInboxBadge();
}

function updateInboxRow(key){
  const el=inboxRowEls[key]; if(!el) return;
  el.innerHTML=inboxRowInner(key);
  el.onclick=()=>selectAction(key);
  el.classList.toggle('sg-sel',selectedId===key);
}

function updateInboxBadge(){
  const cnt=document.getElementById('grpc-inbox');
  if(cnt) cnt.textContent=pending+' / '+inboxTotal;
}

// Sidebar
function ensureGroup(ll){
  if(groups[ll]) return;
  groups[ll]=[]; lifelines.push(ll);
  const em=sidebar.querySelector('.sb-empty'); if(em) em.remove();
  const grp=document.createElement('div');
  grp.className='sg-group'; grp.id='grp-'+ll;
  grp.innerHTML='<div class="sg-hdr"><span><span class="sg-chevron"></span>'+esc(ll)+'</span><span class="sg-count" id="grpc-'+esc(ll)+'">0</span></div>';
  grp.querySelector('.sg-hdr').addEventListener('click', function(){ toggleGroup(grp); });
  sidebar.appendChild(grp);
}

function rowInner(a){
  const req=a.reqId?reqMap.get(a.reqId):null;
  const hp=req&&!req.resolved&&a.kind==='human';
  const dc=hp?'dot-pending':(a.status==='pending'?'dot-running':'dot-done');
  return '<span class="sg-dot '+dc+'"></span><span class="sg-name">'+esc(a.name)+'</span>'+(hp?'<span class="sg-you">you</span>':'');
}

function createRow(key){
  const a=byKey[key];
  const el=document.createElement('div');
  el.className='sg-row'; el.id='row-'+key; el.tabIndex=0;
  el.innerHTML=rowInner(a);
  el.onclick=()=>selectAction(key);
  el.onkeydown=e=>{ if(e.key==='Enter'||e.key===' '){ e.preventDefault(); selectAction(key); } };
  rowEls[key]=el;
  const grp=document.getElementById('grp-'+a.lifeline);
  if(grp){
    grp.appendChild(el);
    const cnt=document.getElementById('grpc-'+a.lifeline);
    if(cnt) cnt.textContent=grp.querySelectorAll('.sg-row').length;
  }
}

function updateRow(key){
  const el=rowEls[key]; if(!el) return;
  el.innerHTML=rowInner(byKey[key]);
  el.onclick=()=>selectAction(key);
  el.classList.toggle('sg-sel',selectedId===key);
}

function selectAction(key){
  if(selectedId===key) return;
  if(selectedId&&rowEls[selectedId]) rowEls[selectedId].classList.remove('sg-sel');
  if(selectedId&&inboxRowEls[selectedId]) inboxRowEls[selectedId].classList.remove('sg-sel');
  selectedId=key;
  if(rowEls[key]) rowEls[key].classList.add('sg-sel');
  if(inboxRowEls[key]) inboxRowEls[key].classList.add('sg-sel');
  renderInspector();
}

// Inspector
function renderInspector(){
  if(!selectedId||!byKey[selectedId]){
    insBody.innerHTML='<div class="ins-empty-state">Click an action to inspect it</div>';
    return;
  }
  const a=byKey[selectedId];
  const req=a.reqId?reqMap.get(a.reqId):null;
  const hp=req&&!req.resolved&&a.kind==='human';
  const hd=req&&req.resolved;
  const kl={llm:'llm',pure:'pure',human:'human',planner:'plan'}[a.kind]||'act';
  const isHuman=a.kind==='human';

  const title=isHuman?(req&&req.instruction?req.instruction:a.name):a.name;
  const isEmailCtx=hp&&req&&req.context&&parseEmailMeta(req.context)!==null;
  // show title for: non-human, done human, confirm (question), or ack (notification)
  const showTitle=!isHuman||hd||(hp&&req&&(req.kind==='confirm'||req.kind==='ack'));
  let html='<div class="ins-meta">'
    +'<span class="ins-meta-ll">'+esc(a.lifeline)+'</span>'
    +'<span class="ins-meta-dot">&middot;</span>'
    +'<span class="ins-meta-kind">'+esc(kl)+'</span>'
    +(isHuman?'<span class="ins-meta-dot">&middot;</span><span class="ins-meta-fn">'+esc(a.name)+'</span>':'')
    +(a.time?'<span class="ins-meta-dot">&middot;</span><span class="ins-meta-time">'+esc(a.time)+'</span>':'')
    +(hp?'<span class="ins-meta-await">awaiting you</span>':'')
    +'</div>';
  if(showTitle) html+='<div class="ins-title">'+esc(title)+'</div>';

  if(hp)      html+=renderPendingForm(req);
  else if(hd) html+=renderHumanDone(req);
  else        html+=renderDoneSection(a);

  insBody.innerHTML=html;
  if(hp) wireInputs(a.reqId,req);
}

function renderCtxHtml(text){
  const lines=text.split('\n');
  return lines.map(function(line){
    const t=line.trim();
    if(!t) return '';
    if(t.endsWith(':')) return '<span class="ctx-hdr">'+esc(t)+'</span>';
    return esc(t);
  }).join('\n');
}

function parseEmailMeta(text){
  if(!text) return null;
  const lines=text.split('\n');
  let from=null, subject=null, bodyStart=-1;
  for(let i=0;i<Math.min(lines.length,6);i++){
    const t=lines[i];
    if(t.startsWith('From: ')) from=t.slice(6).trim();
    else if(t.startsWith('Subject: ')) subject=t.slice(9).trim();
    else if(!t.trim()&&(from||subject)){ bodyStart=i+1; break; }
  }
  if(!from&&!subject) return null;
  if(bodyStart<0) bodyStart=2;
  return {from, subject, body:lines.slice(bodyStart).join('\n').trim()};
}

function renderEmailCtx(text){
  const meta=parseEmailMeta(text);
  if(!meta) return '<div class="ins-ctx">'+renderCtxHtml(text)+'</div>';
  const fromAddrM=meta.from?(meta.from.match(/<([^>]+)>/)||null):null;
  const fromAddr=fromAddrM?fromAddrM[1]:null;
  const fromName=meta.from?(meta.from.replace(/\s*<[^>]*>/,'').trim()||meta.from):null;
  let h='<div class="ea ea-read">';
  if(meta.subject) h+='<div class="ea-subj">'+esc(meta.subject)+'</div>';
  if(fromName){
    const nameSpan=fromAddr
      ?'<span title="'+esc(fromAddr)+'" style="cursor:default">'+esc(fromName)+'</span>'
      :esc(fromName);
    h+='<div class="ea-hdr"><span style="color:var(--text-faint)">From</span> '+nameSpan+'</div>';
  }
  h+='<hr class="ea-rule"><div class="ea-body">'+esc(meta.body||'')+'</div>';
  h+='</div>';
  return h;
}

function renderPendingForm(req){
  const submitLabel=req.submit_label||(req.kind==='confirm'?'Accept':req.kind==='ack'?'Noted':'Approve & send →');
  const cancelLabel=req.cancel_label||'Decline';
  const instruction=req.instruction||'';
  const taVal=esc(req.prefill||'');
  let h='';
  if(req.kind==='ack'){
    if(req.context) h+='<div class="ins-section">'+renderEmailCtx(req.context)+'</div>';
    h+='<div class="ins-actions"><button class="btn-approve" disabled>'+esc(submitLabel)+'</button></div>';
  } else if(req.kind==='confirm'){
    if(req.context) h+='<div class="ins-section">'+renderEmailCtx(req.context)+'</div>';
    h+='<div class="ins-actions"><button class="btn-approve" disabled>'+esc(submitLabel)+'</button>'
      +'<button class="btn-secondary" disabled>'+esc(cancelLabel)+'</button></div>';
  } else {
    const actHtml='<div class="ins-actions"><button class="btn-approve" disabled>'+esc(submitLabel)+'</button>'
      +'<button class="btn-secondary" disabled>'+esc(cancelLabel)+'</button>'
      +'<span class="ins-hint">⌘↩ to approve</span></div>';
    if(req.context){
      const instrHdr=instruction
        ?'<div class="ea-instr">'+esc(instruction)+'</div><hr class="ea-rule">'
        :'';
      h+='<div class="ins-split">'
        +'<div>'+renderEmailCtx(req.context)+'</div>'
        +'<div class="ins-split-work">'
        +'<div class="ea ea-write">'+instrHdr+'<textarea class="ea-ta">'+taVal+'</textarea></div>'
        +actHtml+'</div>'
        +'</div>';
    } else {
      const taLabelCls=instruction?'ins-instr-label':'ins-sec-label';
      const taLabel=instruction||(req.prefill?'Edit or approve':'Input');
      h+='<div class="ins-section"><div class="'+taLabelCls+'">'+esc(taLabel)+'</div>'
        +'<textarea class="ins-ta">'+taVal+'</textarea></div>'+actHtml;
    }
  }
  return h;
}

function renderHumanDone(req){
  const shown=req.declined?'(declined)':req.value===''?'(approved as-is)':req.value;
  return '<div class="ins-section"><div class="ins-sec-label">Response</div>'
    +'<div class="ins-ctx">'+esc(shown)+'</div></div>';
}

function ctxBox(v){
  const val=fmtV(v);
  if(typeof v==='boolean'){
    const cls=v?'kv-val-true':'kv-val-false';
    return '<div class="ins-ctx ins-ctx-mono"><span class="'+cls+'">'+esc(val)+'</span></div>';
  }
  return '<div class="ins-ctx">'+esc(val)+'</div>';
}

function renderDoneSection(a){
  const inE=Object.entries(a.inputs||{}).filter(([,v])=>!isCtrl(v)&&fmtV(v));
  const outE=Object.entries(a.outputs||{}).filter(([,v])=>!isCtrl(v));
  let h='';
  inE.forEach(function([k,v]){
    h+='<div class="ins-section"><div class="ins-sec-label">'+esc(k)+'</div>'+ctxBox(v)+'</div>';
  });
  if(outE.length){
    outE.forEach(function([k,v]){
      h+='<div class="ins-section"><div class="ins-sec-label">'+esc(k)+'</div>'+ctxBox(v)+'</div>';
    });
  } else {
    h+='<div class="ins-section"><div class="ins-sec-label">output</div>'
      +'<div class="ins-ctx"><span class="kv-empty">(empty)</span></div></div>';
  }
  return h;
}

// Wire inputs
let _cmdHandler=null;
function wireInputs(req_id,req){
  if(_cmdHandler){ document.removeEventListener('keydown',_cmdHandler); _cmdHandler=null; }
  if(req.kind==='ack'||req.kind==='confirm'){
    const yes=insBody.querySelector('.btn-approve'),no=insBody.querySelector('.btn-secondary');
    setTimeout(function(){ yes.disabled=false; if(no) no.disabled=false; },600);
    yes.onclick=function(){ doSubmit(req_id,'true'); };
    if(no) no.onclick=function(){ doSubmit(req_id,'false'); };
    _cmdHandler=function(e){
      if((e.metaKey||e.ctrlKey)&&e.key==='Enter'&&!yes.disabled){ e.preventDefault(); doSubmit(req_id,'true'); }
    };
    document.addEventListener('keydown',_cmdHandler);
  } else {
    const ta=insBody.querySelector('.ea-ta,.ins-ta'),btn=insBody.querySelector('.btn-approve');
    const dec=insBody.querySelector('.btn-secondary');
    setTimeout(function(){ btn.disabled=false; if(dec) dec.disabled=false; },800);
    setTimeout(function(){ ta.focus({preventScroll:true}); },900);
    btn.onclick=function(){ doSubmit(req_id,ta.value); };
    if(dec) dec.onclick=function(){ const r=reqMap.get(req_id); if(r) r.declined=true; doSubmit(req_id,''); };
    _cmdHandler=function(e){
      if((e.metaKey||e.ctrlKey)&&e.key==='Enter'&&!btn.disabled){ e.preventDefault(); doSubmit(req_id,ta.value); }
    };
    document.addEventListener('keydown',_cmdHandler);
  }
}

function doSubmit(req_id,val){
  fetch('/human-input',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:req_id,value:val})});
}

// Event handlers
function handleInit(e){
  Object.keys(byKey).forEach(function(k){ delete byKey[k]; });
  Object.keys(groups).forEach(function(k){ delete groups[k]; });
  Object.keys(rowEls).forEach(function(k){ delete rowEls[k]; });
  Object.keys(inboxRowEls).forEach(function(k){ delete inboxRowEls[k]; });
  lifelines.length=0; reqMap.clear(); pending=0; inboxTotal=0; selectedId=null;
  sidebar.innerHTML='<p class="sb-empty">Awaiting workflow…</p>';
  insBody.innerHTML='<div class="ins-empty-state">Click an action to inspect it</div>';
  refreshCount();
  (e.lifelines||[]).forEach(function(ll){ ensureGroup(ll); });
}

function handleRunStart(e){ (e.lifelines||[]).forEach(function(ll){ ensureGroup(ll); }); }

function handleActStart(e){
  const ll=e.lifeline; ensureGroup(ll);
  const kind=e.action_kind||'pure', name=e.action||'—';
  if(name==='assign') return;
  const key=ll+':'+e.seq;
  const now=new Date();
  const time=now.getHours().toString().padStart(2,'0')+':'+now.getMinutes().toString().padStart(2,'0');
  byKey[key]={key:key,lifeline:ll,name:name,kind:kind,seq:e.seq,status:'pending',inputs:{},outputs:{},reqId:null,time:time};
  groups[ll].push(key);
  createRow(key);
  if(kind==='human') createInboxRow(key);
}

function handleAct(e){
  const key=e.lifeline+':'+e.seq;
  const a=byKey[key]; if(!a) return;
  a.status='done'; a.inputs=e.inputs||{}; a.outputs=e.outputs||{};
  updateRow(key);
  if(selectedId===key) renderInspector();
}

function handleHumanRequired(e){
  pending++; refreshCount();
  reqMap.set(e.id,{lifeline:e.lifeline,kind:e.kind||'confirm',context:e.context??null,instruction:e.instruction??null,prefill:e.prefill??null,submit_label:e.submit_label??null,cancel_label:e.cancel_label??null,resolved:false,value:null});
  const llKeys=groups[e.lifeline]||[];
  let matchedKey=null;
  for(let i=llKeys.length-1;i>=0;i--){
    const k=llKeys[i],a=byKey[k];
    if(a&&a.kind==='human'&&a.status==='pending'&&!a.reqId){
      a.reqId=e.id; matchedKey=k; updateRow(k);
      // Only jump to the new action if we're not already looking at a pending human action
      const curReq=selectedId&&byKey[selectedId]&&byKey[selectedId].reqId?reqMap.get(byKey[selectedId].reqId):null;
      if(!curReq||curReq.resolved) selectAction(k);
      break;
    }
  }
  if(matchedKey) updateInboxRow(matchedKey);
}

function handleHumanInput(e){
  const req=reqMap.get(e.id);
  if(req){ req.resolved=true; req.value=e.value; }
  pending=Math.max(0,pending-1); refreshCount();
  Object.keys(byKey).forEach(function(k){ if(byKey[k].reqId===e.id){ updateRow(k); updateInboxRow(k); } });
  if(selectedId&&byKey[selectedId]&&byKey[selectedId].reqId===e.id) renderInspector();
  const entries=[...reqMap];
  for(let i=entries.length-1;i>=0;i--){
    const [id,r]=entries[i];
    if(!r.resolved){
      setTimeout(function(){
        Object.keys(byKey).forEach(function(k){ if(byKey[k].reqId===id){ selectAction(k); } });
      },350);
      return;
    }
  }
}

// Dispatcher
function dispatch(e){
  if(!e||!e.type) return;
  switch(e.type){
    case 'init':                 handleInit(e); break;
    case 'run_start':            handleRunStart(e); break;
    case 'act_start':            handleActStart(e); break;
    case 'act':                  handleAct(e); break;
    case 'human_input_required': handleHumanRequired(e); break;
    case 'human_input':          handleHumanInput(e); break;
  }
}


// SSE
function connect(){
  if(evSrc){ evSrc.close(); evSrc=null; }
  const src=new EventSource('/events');
  evSrc=src;
  src.onmessage=function(ev){ try{ dispatch(JSON.parse(ev.data)); }catch(err){ console.error(err,ev.data); } };
}
connect();

// Keyboard nav
document.addEventListener('keydown',function(e){
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT') return;
  if(e.key!=='ArrowUp'&&e.key!=='ArrowDown') return;
  e.preventDefault();
  const all=[];
  lifelines.forEach(function(ll){ (groups[ll]||[]).forEach(function(k){ if(byKey[k]&&byKey[k].name!=='assign') all.push(k); }); });
  if(!all.length) return;
  const idx=all.indexOf(selectedId);
  const next=e.key==='ArrowDown'?all[Math.min(idx+1,all.length-1)]:all[Math.max(idx-1,0)];
  if(next&&next!==selectedId){ selectAction(next); if(rowEls[next]) rowEls[next].scrollIntoView({block:'nearest'}); }
});
</script>
</body>
</html>
"""
