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

    def __init__(self, lifelines, port: int = 8765, *, dashboard: bool = False, name: str = "", show_decisions: bool = True):
        self._dashboard = dashboard
        self._lifelines = _lifeline_names(lifelines)
        self._port = port
        self._name = name
        self._show_decisions = show_decisions
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
        ev: dict = {"type": "init", "lifelines": self._lifelines, "show_decisions": self._show_decisions}
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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:              #f8fafb;
  --panel:           #f8fafb;
  --text:            #14141A;
  --text-soft:       #3a3a5a;
  --text-mute:       #6e6e92;
  --text-faint:      #a8a8c4;
  --accent:          #1d4ed8;
  --accent-bg:       #dde6fb;
  --accent-attn:     #E94F2E;
  --accent-attn-bg:  #fde8e3;
  --btn-bg:          #14141A;
  --btn-text:        #F5F2EC;
  --done-clr:        #9aaa2a;
  --sans:       'Inter', system-ui, sans-serif;
  --mono:       'JetBrains Mono', ui-monospace, monospace;
  --lifeline-color: #1e40af;
  --col-op-fill:    #d8e0fb;
  --col-op-bdr:     #9aa9ec;
  --col-send-fill:  #cdeadb;
  --col-send-bdr:   #7fc4a6;
  --col-recv-fill:  #ddf3eb;
  --col-recv-bdr:   #a8d5c4;
  --col-sel:        #2f9168;
  --rule:         #e0e4e8;
  --rule-hdr:     #e0e4e8;
  --inbox-bg:     #eef1f4;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body { font-family: var(--sans); background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.4; }

/* ── App shell ──────────────────────────────────────────────────────────── */
#app  { display: flex; flex-direction: column; height: 100vh; }
#hdr  {
  display: flex; align-items: center; gap: 20px;
  padding: 0 24px 0 25px; height: 70px; flex-shrink: 0;
  border-bottom: 1px solid var(--rule-hdr); background: var(--bg);
  position: relative; z-index: 10;
}
#main { display: flex; flex-direction: row; flex: 1; min-height: 0; overflow: hidden; }

/* ── Header ─────────────────────────────────────────────────────────────── */
.hdr-logo { height: 36px; display: block; }
.hdr-right { display: flex; align-items: center; gap: 8px; margin-left: auto; }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

/* ── Inbox panel ─────────────────────────────────────────────────────────── */
#inbox-panel {
  flex: 0 0 250px; display: flex; flex-direction: column;
  border-right: 1px solid var(--rule); overflow: hidden;
  position: relative; z-index: 10; background: var(--inbox-bg);
  transition: flex-basis 0.22s ease;
}
#inbox-panel.inbox-collapsed { flex-basis: 52px; }
#inbox-panel.inbox-collapsed .inbox-hdr,
#inbox-panel.inbox-collapsed #inbox-list { display: none; }
.inbox-strip {
  display: none; flex-direction: column; align-items: center;
  padding: 8px 0 0; gap: 8px; flex: 1;
}
#inbox-panel.inbox-collapsed .inbox-strip { display: flex; }
.inbox-strip-btn {
  background: none; border: none; cursor: pointer; border-radius: 4px;
  font-size: 15px; color: var(--text-mute); padding: 3px 5px; line-height: 1;
  transition: color .12s;
}
.inbox-strip-btn:hover { color: var(--text); background: rgba(0,0,0,.06); }
.inbox-strip-icon { color: var(--text-mute); display: block; }
.inbox-strip-count {
  min-width: 18px; height: 18px; padding: 0 4px;
  border-radius: 99px; background: var(--accent-attn);
  color: #fff; font-size: 10px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
}
.inbox-strip-count:empty { display: none; }
.inbox-hdr {
  display: flex; align-items: center;
  padding: 10px 12px 10px 0; font-size: 13px; font-weight: 600;
  color: var(--text); border-bottom: 1px solid var(--rule-hdr); flex-shrink: 0;
  letter-spacing: 0.02em;
}
.inbox-hdr-gutter {
  flex: 0 0 24px; display: flex; align-items: center; justify-content: center;
}
.inbox-hdr-title { display: flex; align-items: baseline; gap: 6px; flex: 1; min-width: 0; }
#inbox-fold {
  width: 22px; height: 22px;
  background: none; border: none; cursor: pointer; border-radius: 4px;
  color: var(--text-mute); padding: 0;
  display: flex; align-items: center; justify-content: center; transition: color .12s;
}
#inbox-fold:hover { color: var(--text); background: rgba(0,0,0,.06); }
.inbox-badge { font-size: 11px; font-weight: 700; color: var(--accent-attn); }
#inbox-list { flex: 1; overflow-y: auto; }
.inbox-empty { padding: 24px 12px 24px 24px; font-size: 13px; color: var(--text-faint); }
.inbox-card {
  padding: 9px 12px 9px 24px; cursor: pointer; user-select: none;
  border-bottom: 1px solid var(--rule); transition: background .1s;
}
.inbox-card:hover:not(.inbox-sel) { background: rgba(0,0,0,.03); }
.inbox-card.inbox-sel { background: var(--accent-bg); }
.inbox-card-top { display: flex; align-items: center; gap: 8px; }
.inbox-card-name {
  font-size: 13px; font-weight: 500; color: var(--text);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;
}
.inbox-card.inbox-sel .inbox-card-name { color: var(--text); }
.inbox-card-sub { font-size: 11px; color: var(--text-mute); margin-top: 3px; }
.inbox-card.inbox-sel .inbox-card-sub { color: var(--text-mute); }
.sg-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.dot-done    { background: transparent; }
.dot-running { background: transparent; }
.dot-pending { background: var(--accent-attn); animation: pulse 1.2s ease-in-out infinite; }

/* ── Right panel ─────────────────────────────────────────────────────────── */
#right-panel {
  flex: 1; min-height: 0; position: relative;
  display: flex; flex-direction: column; overflow: hidden;
}

/* ── Column view ─────────────────────────────────────────────────────────── */
#col-view {
  flex: 1; min-height: 0;
  display: flex; flex-direction: row;
  overflow: auto;
  padding: 0 25px 0 0;
  align-items: flex-start;
}
.col-empty { padding: 40px 28px; font-size: 13px; color: var(--text-faint); }
.col-lifeline {
  flex: 0 0 250px; display: flex; flex-direction: column;
  border-right: 1px solid var(--rule); min-height: 100%;
}
.col-lifeline:first-child { border-left: 1px solid var(--rule); }
.col-hdr {
  padding: 14px 16px 10px; font-size: 13px; font-weight: 600;
  color: var(--text); border-bottom: 1px solid var(--rule-hdr);
  flex-shrink: 0; letter-spacing: 0.02em; text-align: center;
  position: sticky; top: 0; background: var(--bg); z-index: 20;
}
.col-content { overflow: visible; padding: 8px; position: relative; }
.col-content .col-card {
  margin-bottom: 0; height: 50px; overflow: hidden;
  display: flex; flex-direction: column; justify-content: flex-start; padding: 6px 10px;
}
.col-card {
  border: 1.5px solid var(--rule); border-radius: 5px;
  padding: 7px 10px; margin-bottom: 7px; cursor: pointer;
  background: var(--panel); transition: border-color .12s, background .12s;
}
.col-card:hover { filter: brightness(.96); }
.col-card.col-sel { border-width: 2px; }
.col-card.col-pending { border-width: 2px; }
.col-card-row { display: flex; align-items: center; gap: 6px; }
.col-card-name {
  font-size: 12px; color: var(--text);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;
}
.col-card-preview {
  font-size: 10px; color: #6b7280; margin-top: 2px;
  font-family: var(--mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.col-msg-arrow { font-size: 11px; color: #4a4f5a; flex-shrink: 0; width: 14px; text-align: center; }
.col-msg-partner { font-size: 12px; color: var(--lifeline-color); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.col-act-llm, .col-act-pure, .col-act-planner { background: var(--col-op-fill); border-color: var(--col-op-bdr); }
.col-act-human { background: var(--accent-attn-bg); border-color: var(--accent-attn); border-width: 1px; }
.col-msg-send { background: var(--col-send-fill); border-color: var(--col-send-bdr); }
.col-msg-recv { background: var(--col-recv-fill); border-color: var(--col-recv-bdr); }
.col-act-llm.col-sel, .col-act-pure.col-sel, .col-act-planner.col-sel { border-color: var(--accent); }
.col-act-human.col-sel   { border-color: #c53a22; }
.col-msg-send.col-sel, .col-msg-recv.col-sel { border-color: var(--col-sel); }

/* ── Detail panel ────────────────────────────────────────────────────────── */
#detail-panel {
  position: absolute; inset: 0; background: var(--bg);
  display: flex; flex-direction: column; z-index: 25;
}
#detail-topbar {
  display: flex; align-items: center; justify-content: flex-end;
  padding: 14px 56px 10px; border-bottom: 1px solid var(--rule-hdr); flex-shrink: 0;
}
#detail-close {
  background: none; border: none; cursor: pointer;
  font-size: 13px; color: var(--text-mute); padding: 3px 8px;
  border-radius: 4px; line-height: 1; font-weight: 500;
}
#detail-close:hover { color: var(--text); background: rgba(0,0,0,.05); }
#detail-body { flex: 1; overflow-y: auto; padding: 32px 56px; }

/* ── Inspector content ───────────────────────────────────────────────────── */
.ins-empty-state {
  display: flex; align-items: center; justify-content: center;
  min-height: 300px; font-size: 14px; color: var(--text-faint);
}
.ins-meta {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  margin-bottom: 12px; font-size: 14px; color: var(--text-mute);
}
.ins-meta-ll   { font-weight: 500; color: var(--lifeline-color); }
.ll-name { color: var(--lifeline-color); }
.ins-arrow { color: #4a4f5a; }
.ins-meta-dot  { color: var(--text-faint); }
.ins-meta-kind { font-size: 12px; }
.ins-meta-time { font-size: 12px; color: var(--text-faint); }
.ins-meta-fn   { font-size: 12px; color: var(--text-faint); }
.ins-await-row { margin: 0 0 8px; }
.ins-meta-await {
  font-family: var(--mono); font-size: 10px; font-weight: 600;
  letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--accent-attn); background: var(--accent-attn-bg);
  border-radius: 3px; padding: 2px 7px;
}
.ins-title {
  font-family: var(--sans); font-size: 24px; font-weight: 600;
  letter-spacing: -0.01em; line-height: 1.2; color: var(--text); margin-bottom: 24px;
}
.ins-section { margin-bottom: 24px; }
.ins-sec-label { font-size: 13px; font-weight: 600; color: var(--text-mute); margin-bottom: 10px; }
.ins-instr-label { font-size: 14px; font-weight: 500; color: var(--text-soft); margin-bottom: 10px; }
.ins-sec-body { font-size: 15px; line-height: 1.6; color: var(--text); }
.ins-ctx {
  white-space: pre-wrap; word-break: break-word; color: var(--text-soft);
  border: 1px solid rgba(20,20,40,0.11); border-radius: 6px; padding: 12px 14px;
  font-size: 15px; line-height: 1.6;
}
.ctx-hdr  { font-weight: 600; color: var(--text); }
.ctx-hint { font-style: italic; color: var(--text-mute); }
.ins-ctx-mono { font-family: var(--mono); font-size: 13px; }
.ea { border-radius: 6px; padding: 20px 18px; }
.ea-read { border: 1px solid rgba(20,20,40,0.11); white-space: pre-wrap; word-break: break-word; }
.ea-write { border: 1px solid rgba(20,20,40,0.13); transition: border-color .15s; }
.ea-write:focus-within { border-color: var(--accent); }
.ea-subj {
  font-family: var(--sans); font-size: 20px; font-weight: 500;
  color: var(--text); line-height: 1.3; margin-bottom: 4px; white-space: normal;
}
.ea-hdr   { font-size: 14px; color: var(--text-mute); }
.ea-instr { font-size: 13px; color: var(--text-mute); }
.ea-rule  { border: none; border-top: 1px solid rgba(20,18,12,0.1); margin: 10px 0 12px; }
.ea-body  { font-size: 15px; line-height: 1.6; color: var(--text-soft); }
.ea-ta {
  width: 100%; font-family: var(--sans); font-size: 15px; line-height: 1.6;
  padding: 0; border: none; background: transparent; color: var(--text);
  resize: none; min-height: 160px; field-sizing: content; outline: none;
}
.ea-ta::placeholder { color: var(--text-faint); }
.ins-resolved-val { color: var(--done-clr); }
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
.ins-ta::placeholder { color: var(--text-faint); }
.ins-ta:focus { border-color: var(--accent); }
.ins-actions { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 20px; }
.btn-approve {
  font-family: var(--sans); font-size: 14px; font-weight: 500;
  padding: 10px 28px; border-radius: 6px;
  background: var(--accent); color: #fff;
  border: none; cursor: pointer; transition: opacity .15s;
}
.btn-approve:hover:not(:disabled) { opacity: .85; }
.btn-approve:disabled { opacity: .35; cursor: default; }
.btn-secondary {
  font-family: var(--sans); font-size: 14px; font-weight: 500;
  padding: 10px 28px; border-radius: 6px;
  background: transparent; color: var(--text-soft);
  border: 1px solid rgba(20,18,12,0.15); cursor: pointer; transition: all .15s;
}
.btn-secondary:hover:not(:disabled) { border-color: rgba(20,18,12,0.3); color: var(--text); }
.btn-secondary:disabled { opacity: .35; cursor: default; }
.ins-hint { text-align: center; margin-top: 12px; font-size: 12px; color: var(--text-faint); }
.btn-choice {
  font-family: var(--sans); font-size: 14px; padding: 9px 16px; border-radius: 6px;
  border: 1px solid var(--rule); background: var(--panel); color: var(--text-soft);
  cursor: pointer; transition: all .12s;
}
.btn-choice:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); background: var(--accent-bg); }
.btn-choice:disabled { opacity: .3; cursor: default; }
.sel-opts { display: flex; flex-direction: column; gap: 8px; margin: 4px 0; }
.sel-opt.sel-active { border-color: var(--accent); color: var(--accent); background: var(--accent-bg); }
.kv-row { display: flex; gap: 14px; padding: 6px 0; border-bottom: 1px solid var(--rule); }
.kv-row:last-child { border-bottom: none; }
.kv-key { font-family: var(--mono); font-size: 12px; color: var(--text-mute); min-width: 100px; flex-shrink: 0; padding-top: 2px; }
.kv-val { font-size: 15px; color: var(--text); line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
.kv-val-true  { font-family: var(--mono); color: var(--done-clr); }
.kv-val-false { font-family: var(--mono); color: var(--text-faint); }
.kv-empty     { font-size: 13px; color: var(--text-faint); }

/* ── Message arrows SVG ──────────────────────────────────────────────────── */
#col-arrows { position: fixed; inset: 0; pointer-events: none; overflow: visible; z-index: 9; color: rgba(20,20,26,.6); }
#col-arrows line { stroke: currentColor; stroke-width: 1.5; marker-end: url(#arr-tip); }
#btn-arrows {
  font-family: var(--sans); font-size: 12px; font-weight: 500;
  padding: 4px 14px; border-radius: 6px; border: 1px solid var(--rule);
  cursor: pointer; background: transparent; color: var(--text-mute); transition: all .12s;
}
#btn-arrows.arr-on { border-color: var(--col-op-bdr); color: var(--text); }
#btn-arrows:hover:not(.arr-on) { color: var(--text-soft); }

/* ── Decision / control-broadcast ────────────────────────────────────────── */
.col-decision {
  position: absolute; left: 28px; right: 28px;
  background: #e5e3c7; border: 1px solid #bcb576; border-radius: 8px;
  height: 50px; overflow: hidden; cursor: pointer;
  display: flex; flex-direction: column; justify-content: center; padding: 0 10px;
  transition: border-color .12s;
}
.col-decision:hover { border-color: #8a8040; }
.col-decision.col-dec-sel { border: 2px solid #8a8040; }
.col-dec-label { font-size: 12px; font-weight: 600; color: #4a4820; }
.col-dec-cond  { font-size: 11px; color: #6b6840; font-family: var(--mono); margin-top: 2px;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.col-ctrl-circle {
  position: absolute; width: 40px; height: 40px; border-radius: 50%;
  background: #e5e3c7; border: 1px solid #bcb576; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 600; color: #4a4820;
  box-shadow: 0 0 0 3px var(--bg); transition: border-color .12s;
}
.col-ctrl-circle:hover { border-color: #8a8040; }

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
    <div class="hdr-right">
      <button id="btn-arrows" onclick="toggleArrows()" title="Show message arrows" style="display:none">arrows</button>
    </div>
  </header>
  <div id="main">
    <div id="inbox-panel">
      <div class="inbox-hdr">
        <div class="inbox-hdr-gutter">
          <button id="inbox-fold" aria-label="Collapse inbox">
            <svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg>
          </button>
        </div>
        <span class="inbox-hdr-title">Inbox<span class="inbox-badge" id="inbox-badge"></span></span>
      </div>
      <div id="inbox-list">
        <p class="inbox-empty">No actions yet&hellip;</p>
      </div>
      <div class="inbox-strip">
        <button class="inbox-strip-btn" onclick="document.getElementById('inbox-fold').click()" aria-label="Expand inbox">
          <svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>
        </button>
        <svg class="inbox-strip-icon" aria-hidden="true" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 12h-6l-2 3h-4l-2-3H2"/>
          <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>
        </svg>
        <span class="inbox-strip-count" id="inbox-strip-count"></span>
      </div>
    </div>
    <div id="right-panel">
      <div id="col-view">
        <p class="col-empty">Awaiting workflow&hellip;</p>
      </div>
      <div id="detail-panel" style="display:none">
        <div id="detail-topbar">
          <button id="detail-close" onclick="closeDetail()">← Back</button>
        </div>
        <div id="detail-body"></div>
      </div>
    </div>
  </div>
  <svg id="col-arrows">
    <defs>
      <marker id="arr-tip" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto">
        <polygon points="0 0, 6 3, 0 6" fill="currentColor"/>
      </marker>
    </defs>
    <g id="col-arrows-g"></g>
  </svg>
</div>
<script>
// ── State ────────────────────────────────────────────────────────────────────
let evSrc        = null;
const byKey      = {};
const groups     = {};
const lifelines  = [];
const inboxCards = {};   // key → inbox card element (human actions only)
const reqMap     = new Map();
let pending      = 0;
let detailKey    = null; // key currently shown in detail panel

// Column view state
const colEls      = {};
const colActCards = {};
const msgCards    = {};
const colRowIdx   = {};
const sendRowIdx  = {};
const COL_PAD = 8;
const COL_GAP = 8;
const ROW_H   = 50;
const decisionRows   = new Set();
let globalColH       = 0;
let showArrows       = false;
let decisionsEnabled = true;
let _arrowRaf        = null;
const currentDecision = {};
const ctrlCards       = {};
const decisionEls     = {};  // 'dec:N' → decision box element
let _ctrlSeq          = 0;

// ── DOM ──────────────────────────────────────────────────────────────────────
const inboxList   = document.getElementById('inbox-list');
const inboxBadge  = document.getElementById('inbox-badge');
const inboxPanel  = document.getElementById('inbox-panel');
const detailPanel = document.getElementById('detail-panel');
const detailBody  = document.getElementById('detail-body');
const colView     = document.getElementById('col-view');
const arrowsGroup = document.getElementById('col-arrows-g');
colView.addEventListener('scroll', function(){ if(showArrows) scheduleDrawArrows(); });

// ── Inbox fold ────────────────────────────────────────────────────────────────
const _FOLD_KEY = 'zc-inbox-collapsed';
const _foldBtn  = document.getElementById('inbox-fold');
function _applyFold(collapsed){
  inboxPanel.classList.toggle('inbox-collapsed', collapsed);
  _foldBtn.title = collapsed ? 'Expand inbox' : 'Collapse inbox';
}
_applyFold(localStorage.getItem(_FOLD_KEY) === '1');
_foldBtn.onclick = function(){
  const collapsed = !inboxPanel.classList.contains('inbox-collapsed');
  _applyFold(collapsed);
  localStorage.setItem(_FOLD_KEY, collapsed ? '1' : '0');
};
inboxPanel.addEventListener('transitionend', function(){ if(showArrows) scheduleDrawArrows(); });

// ── Message arrows ────────────────────────────────────────────────────────────
function scheduleDrawArrows(){
  if(_arrowRaf) cancelAnimationFrame(_arrowRaf);
  _arrowRaf = requestAnimationFrame(drawArrows);
}

function drawArrows(){
  _arrowRaf = null;
  if(!showArrows){ arrowsGroup.innerHTML=''; return; }
  const vr = colView.getBoundingClientRect();
  let lines = '';
  Object.values(msgCards).forEach(function(pair){
    const sEl=pair.send, rEl=pair.recv;
    if(!sEl||!rEl) return;
    const sRect=sEl.getBoundingClientRect(), rRect=rEl.getBoundingClientRect();
    if(sRect.bottom<vr.top||sRect.top>vr.bottom) return;
    const sCx=(sRect.left+sRect.right)/2, rCx=(rRect.left+rRect.right)/2;
    const x1=rCx>=sCx?sRect.right:sRect.left, x2=rCx>=sCx?rRect.left:rRect.right;
    const y1=(sRect.top+sRect.bottom)/2, y2=(rRect.top+rRect.bottom)/2;
    lines+='<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'"/>';
  });
  arrowsGroup.innerHTML = lines;
}

function toggleArrows(){
  showArrows = !showArrows;
  document.getElementById('btn-arrows').classList.toggle('arr-on', showArrows);
  drawArrows();
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function esc(s){ return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function isCtrl(v){ return typeof v==='string'&&v.startsWith('κ_ctrl_'); }
function fmtV(v){
  if(v===null||v===undefined) return '';
  if(isCtrl(v)) return '';
  if(typeof v==='boolean') return v?'true':'false';
  if(typeof v==='string') return v.length>600?v.slice(0,599)+'…':v;
  try{ return JSON.stringify(v,null,2); }catch{ return String(v); }
}

// ── Inbox ─────────────────────────────────────────────────────────────────────
function inboxCardInner(key){
  const a = byKey[key];
  const req = a.reqId ? reqMap.get(a.reqId) : null;
  const hp = req && !req.resolved && a.kind==='human';
  const sub = a.lifeline + (a.time ? ' · ' + a.time : '');
  return '<div class="inbox-card-top">'
    +'<span class="inbox-card-name">'+esc(a.name)+'</span>'
    +(hp?'<span class="sg-dot dot-pending"></span>':'')
    +'</div>'
    +'<div class="inbox-card-sub">'+esc(sub)+'</div>';
}

function createInboxCard(key){
  const em = inboxList.querySelector('.inbox-empty'); if(em) em.remove();
  const el = document.createElement('div');
  el.className = 'inbox-card'; el.dataset.key = key;
  el.innerHTML = inboxCardInner(key);
  el.onclick = function(){ openDetail(key); };
  inboxCards[key] = el;
  inboxList.insertBefore(el, inboxList.firstChild);
  updateInboxBadge();
}

function updateInboxCard(key){
  const el = inboxCards[key]; if(!el) return;
  el.innerHTML = inboxCardInner(key);
  el.onclick = function(){ openDetail(key); };
  el.classList.toggle('inbox-sel', detailKey===key);
}

function updateInboxBadge(){
  const t = pending > 0 ? pending : '';
  inboxBadge.textContent = t;
  document.getElementById('inbox-strip-count').textContent = t;
}

// ── Detail panel ──────────────────────────────────────────────────────────────
function _clearDetailSel(key){
  if(!key) return;
  if(colActCards[key]) colActCards[key].classList.remove('col-sel');
  const mp = msgCards[key];
  if(mp){ if(mp.send) mp.send.classList.remove('col-sel'); if(mp.recv) mp.recv.classList.remove('col-sel'); }
  if(decisionEls[key]) decisionEls[key].classList.remove('col-dec-sel');
}

function openDetail(key){
  _clearDetailSel(detailKey);
  detailKey = key;
  Object.keys(inboxCards).forEach(function(k){ inboxCards[k].classList.toggle('inbox-sel', k===key); });
  if(colActCards[key]) colActCards[key].classList.add('col-sel');
  const mp = msgCards[key];
  if(mp){ if(mp.send) mp.send.classList.add('col-sel'); if(mp.recv) mp.recv.classList.add('col-sel'); }
  if(decisionEls[key]) decisionEls[key].classList.add('col-dec-sel');
  document.getElementById('btn-arrows').style.display = 'none';
  detailPanel.style.display = 'flex';
  renderInspector(key, detailBody, closeDetail);
}

function closeDetail(){
  _clearDetailSel(detailKey);
  Object.keys(inboxCards).forEach(function(k){ inboxCards[k].classList.remove('inbox-sel'); });
  detailKey = null;
  detailPanel.style.display = 'none';
  if(Object.keys(colEls).length) document.getElementById('btn-arrows').style.display = '';
  if(_cmdHandler){ document.removeEventListener('keydown',_cmdHandler); _cmdHandler=null; }
}

// ── Inspector ─────────────────────────────────────────────────────────────────
function renderMsgDetail(a){
  const vars = Object.entries(a.bindings||{}).filter(function([k,v]){
    return k!=='branch'&&k!=='loop'&&!isCtrl(v);
  });
  let h = '<div class="ins-meta">'
    +'<span class="ins-meta-ll">'+esc(a.from)+'</span>'
    +'<span class="ins-meta-dot">&#x2192;</span>'
    +'<span class="ins-meta-ll">'+esc(a.to)+'</span>'
    +(a.channel?'<span class="ins-meta-dot">&middot;</span><span class="ins-meta-kind">'+esc(a.channel)+'</span>':'')
    +'</div>'
    +'<div class="ins-title">'+esc(a.from)+' → '+esc(a.to)+'</div>';
  if(vars.length){
    vars.forEach(function([k,v]){
      h += '<div class="ins-section"><div class="ins-sec-label">'+esc(k)+'</div>'+ctxBox(v)+'</div>';
    });
  } else {
    h += '<div class="ins-section"><div class="ins-sec-label">variables</div>'
      +'<div class="ins-ctx"><span class="kv-empty">(none)</span></div></div>';
  }
  return h;
}

function renderDecisionDetail(a){
  const sym = a.value ? '⊤' : '⊥';
  let h = '<div class="ins-meta">'
    +'<span class="ins-meta-ll">'+esc(a.lifeline)+'</span>'
    +'<span class="ins-meta-dot">&middot;</span>'
    +'<span class="ins-meta-kind">'+esc(a.decKind)+'</span>'
    +'</div>'
    +'<div class="ins-title">'+esc(a.decKind)+' '+sym+'</div>';
  const formula = a.formula||a.condition||'';
  if(formula){
    h += '<div class="ins-section"><div class="ins-sec-label">condition</div>'
      +'<div class="ins-ctx ins-ctx-mono">'+esc(formula)+'</div></div>';
  }
  h += '<div class="ins-section"><div class="ins-sec-label">value</div>'
    +'<div class="ins-ctx ins-ctx-mono"><span class="'+(a.value?'kv-val-true':'kv-val-false')+'">'+esc(sym)+'</span></div></div>';
  return h;
}

function renderInspector(overrideKey, targetEl, afterInput){
  const key = overrideKey !== undefined ? overrideKey : detailKey;
  const el  = targetEl || detailBody;
  if(!key||!byKey[key]){
    el.innerHTML = '<div class="ins-empty-state">Click an action to inspect it</div>';
    return;
  }
  const a   = byKey[key];
  if(a.kind==='msg'){      el.innerHTML = renderMsgDetail(a);      return; }
  if(a.kind==='decision'){ el.innerHTML = renderDecisionDetail(a); return; }
  const req = a.reqId ? reqMap.get(a.reqId) : null;
  const hp  = req && !req.resolved && a.kind==='human';
  const hd  = req && req.resolved;
  const kl  = {llm:'llm',pure:'pure',human:'human',planner:'plan'}[a.kind]||'act';
  const isHuman = a.kind==='human';
  const title = isHuman ? (req&&req.instruction ? req.instruction : a.name) : a.name;
  const showTitle = !isHuman||hd||(hp&&req&&(req.kind==='confirm'||req.kind==='ack'));
  let html = '<div class="ins-meta">'
    +'<span class="ins-meta-ll">'+esc(a.lifeline)+'</span>'
    +'<span class="ins-meta-dot">&middot;</span>'
    +'<span class="ins-meta-kind">'+esc(kl)+'</span>'
    +(isHuman?'<span class="ins-meta-dot">&middot;</span><span class="ins-meta-fn">'+esc(a.name)+'</span>':'')
    +(a.time?'<span class="ins-meta-dot">&middot;</span><span class="ins-meta-time">'+esc(a.time)+'</span>':'')
    +'</div>'
    +(hp?'<div class="ins-await-row"><span class="ins-meta-await">AWAITING YOU</span></div>':'');
  if(showTitle) html += '<div class="ins-title">'+esc(title)+'</div>';
  if(hp)       html += renderPendingForm(req);
  else if(hd)  html += renderHumanDone(req);
  else         html += renderDoneSection(a);
  el.innerHTML = html;
  if(hp) wireInputs(a.reqId, req, afterInput||null, el);
}

function renderCtxHtml(text){
  return text.split('\n').map(function(line){
    const t = line.trim();
    if(!t) return '';
    if(t.endsWith(':')) return '<span class="ctx-hdr">'+esc(t)+'</span>';
    return esc(t);
  }).join('\n');
}

function parseEmailMeta(text){
  if(!text) return null;
  const lines = text.split('\n');
  let from=null, subject=null, bodyStart=-1;
  for(let i=0;i<Math.min(lines.length,6);i++){
    const t = lines[i];
    if(t.startsWith('From: ')) from = t.slice(6).trim();
    else if(t.startsWith('Subject: ')) subject = t.slice(9).trim();
    else if(!t.trim()&&(from||subject)){ bodyStart=i+1; break; }
  }
  if(!from&&!subject) return null;
  if(bodyStart<0) bodyStart=2;
  return {from, subject, body:lines.slice(bodyStart).join('\n').trim()};
}

function renderEmailCtx(text){
  const meta = parseEmailMeta(text);
  if(!meta) return '<div class="ins-ctx">'+renderCtxHtml(text)+'</div>';
  const fromAddrM = meta.from ? (meta.from.match(/<([^>]+)>/)||null) : null;
  const fromAddr  = fromAddrM ? fromAddrM[1] : null;
  const fromName  = meta.from ? (meta.from.replace(/\s*<[^>]*>/,'').trim()||meta.from) : null;
  let h = '<div class="ea ea-read">';
  if(meta.subject) h += '<div class="ea-subj">'+esc(meta.subject)+'</div>';
  if(fromName){
    const nameSpan = fromAddr
      ? '<span title="'+esc(fromAddr)+'" style="cursor:default">'+esc(fromName)+'</span>'
      : esc(fromName);
    h += '<div class="ea-hdr"><span style="color:var(--text-faint)">From</span> '+nameSpan+'</div>';
  }
  h += '<hr class="ea-rule"><div class="ea-body">'+esc(meta.body||'')+'</div></div>';
  return h;
}

function renderPendingForm(req){
  const submitLabel = req.submit_label||(req.kind==='confirm'?'Accept':req.kind==='ack'?'Noted':'Approve & send →');
  const cancelLabel = req.cancel_label||'Decline';
  const instruction = req.instruction||'';
  const taVal = esc(req.prefill||'');
  let h = '';
  if(req.kind==='ack'){
    if(req.context) h += '<div class="ins-section">'+renderEmailCtx(req.context)+'</div>';
    h += '<div class="ins-actions"><button class="btn-approve" disabled>'+esc(submitLabel)+'</button></div>';
  } else if(req.kind==='confirm'){
    if(req.context) h += '<div class="ins-section">'+renderEmailCtx(req.context)+'</div>';
    h += '<div class="ins-actions"><button class="btn-approve" disabled>'+esc(submitLabel)+'</button>'
      +'<button class="btn-secondary" disabled>'+esc(cancelLabel)+'</button></div>'
      +'<div class="ins-hint">⌘↩ to confirm</div>';
  } else {
    const actHtml = '<div class="ins-actions"><button class="btn-approve" disabled>'+esc(submitLabel)+'</button>'
      +'<button class="btn-secondary" disabled>'+esc(cancelLabel)+'</button></div>'
      +'<div class="ins-hint">⌘↩ to approve</div>';
    if(req.context){
      const instrHdr = instruction
        ? '<div class="ea-instr">'+esc(instruction)+'</div><hr class="ea-rule">'
        : '';
      h += '<div class="ins-split">'
        +'<div>'+renderEmailCtx(req.context)+'</div>'
        +'<div class="ins-split-work">'
        +'<div class="ea ea-write">'+instrHdr+'<textarea class="ea-ta">'+taVal+'</textarea></div>'
        +'</div></div>'
        +actHtml;
    } else {
      const taLabelCls = instruction ? 'ins-instr-label' : 'ins-sec-label';
      const taLabel = instruction||(req.prefill?'Edit or approve':'Input');
      h += '<div class="ins-section"><div class="'+taLabelCls+'">'+esc(taLabel)+'</div>'
        +'<textarea class="ins-ta">'+taVal+'</textarea></div>'+actHtml;
    }
  }
  return h;
}

function renderHumanDone(req){
  const shown = req.declined ? '(declined)' : req.value==='' ? '(approved as-is)' : req.value;
  return '<div class="ins-section"><div class="ins-sec-label">Response</div>'
    +'<div class="ins-ctx">'+esc(shown)+'</div></div>';
}

function ctxBox(v){
  const val = fmtV(v);
  if(typeof v==='boolean'){
    const cls = v ? 'kv-val-true' : 'kv-val-false';
    return '<div class="ins-ctx ins-ctx-mono"><span class="'+cls+'">'+esc(val)+'</span></div>';
  }
  return '<div class="ins-ctx">'+esc(val)+'</div>';
}

function renderDoneSection(a){
  const inE  = Object.entries(a.inputs||{}).filter(([,v])=>!isCtrl(v)&&fmtV(v));
  const outE = Object.entries(a.outputs||{}).filter(([,v])=>!isCtrl(v));
  let h = '';
  inE.forEach(function([k,v]){
    h += '<div class="ins-section"><div class="ins-sec-label">'+esc(k)+'</div>'+ctxBox(v)+'</div>';
  });
  if(outE.length){
    outE.forEach(function([k,v]){
      h += '<div class="ins-section"><div class="ins-sec-label">'+esc(k)+'</div>'+ctxBox(v)+'</div>';
    });
  } else {
    h += '<div class="ins-section"><div class="ins-sec-label">output</div>'
      +'<div class="ins-ctx"><span class="kv-empty">(empty)</span></div></div>';
  }
  return h;
}

// ── Wire inputs ───────────────────────────────────────────────────────────────
let _cmdHandler = null;
function wireInputs(req_id, req, afterInput, containerEl){
  containerEl = containerEl || detailBody;
  if(_cmdHandler){ document.removeEventListener('keydown',_cmdHandler); _cmdHandler=null; }
  const sub = function(val){ doSubmit(req_id,val); if(afterInput) afterInput(); };
  if(req.kind==='ack'||req.kind==='confirm'){
    const yes=containerEl.querySelector('.btn-approve'), no=containerEl.querySelector('.btn-secondary');
    setTimeout(function(){ yes.disabled=false; if(no) no.disabled=false; }, 600);
    yes.onclick = function(){ sub('true'); };
    if(no) no.onclick = function(){ sub('false'); };
    _cmdHandler = function(e){
      if((e.metaKey||e.ctrlKey)&&e.key==='Enter'&&!yes.disabled){ e.preventDefault(); sub('true'); }
    };
    document.addEventListener('keydown', _cmdHandler);
  } else {
    const ta=containerEl.querySelector('.ea-ta,.ins-ta'), btn=containerEl.querySelector('.btn-approve');
    const dec = containerEl.querySelector('.btn-secondary');
    setTimeout(function(){ btn.disabled=false; if(dec) dec.disabled=false; }, 800);
    setTimeout(function(){ ta.focus({preventScroll:true}); }, 900);
    btn.onclick = function(){ sub(ta.value); };
    if(dec) dec.onclick = function(){ const r=reqMap.get(req_id); if(r) r.declined=true; sub(''); };
    _cmdHandler = function(e){
      if((e.metaKey||e.ctrlKey)&&e.key==='Enter'&&!btn.disabled){ e.preventDefault(); sub(ta.value); }
    };
    document.addEventListener('keydown', _cmdHandler);
  }
}

function doSubmit(req_id, val){
  fetch('/human-input',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:req_id,value:val})});
}

// ── Column view ───────────────────────────────────────────────────────────────
function msgKey(e){ return e.from+'→'+e.to+'@'+(e.channel||'')+'#'+e.seq; }
function isCtrlMsg(e){ return e.ctrl || (e.values||[]).some(isCtrl); }
function rowToY(row){ return COL_PAD + row*(ROW_H+COL_GAP); }
function colNextRow(ll){
  let r = colRowIdx[ll]!==undefined ? colRowIdx[ll] : 0;
  while(decisionRows.has(r)) r++;
  return r;
}

function colSetScrollH(ll, h){
  globalColH = Math.max(globalColH, h);
  Object.keys(colEls).forEach(function(k){
    const s = document.getElementById('col-spacer-'+k);
    if(s) s.style.height = globalColH+'px';
  });
}

function colPlace(el, ll, row){
  const y = rowToY(row);
  el.style.position = 'absolute';
  el.style.top = y+'px'; el.style.left = '28px'; el.style.right = '28px';
  colEls[ll].appendChild(el);
  colRowIdx[ll] = row+1;
  colSetScrollH(ll, y+ROW_H+COL_PAD);
  if(showArrows) scheduleDrawArrows();
}

function ensureColGroup(ll){
  if(colEls[ll]) return;
  const em = colView.querySelector('.col-empty'); if(em) em.remove();
  document.getElementById('btn-arrows').style.display = '';
  const col = document.createElement('div');
  col.className = 'col-lifeline';
  col.innerHTML = '<div class="col-hdr">'+esc(ll)+'</div>'
    +'<div class="col-content" id="col-content-'+esc(ll)+'"></div>';
  colView.appendChild(col);
  colEls[ll] = col.querySelector('.col-content');
  const spacer = document.createElement('div');
  spacer.id = 'col-spacer-'+ll; spacer.style.height = '0';
  colEls[ll].appendChild(spacer);
}

function ensureGroup(ll){
  if(groups[ll]) return;
  groups[ll] = []; lifelines.push(ll);
}

function colActCardInner(a){
  const req = a.reqId ? reqMap.get(a.reqId) : null;
  const hp  = req && !req.resolved && a.kind==='human';
  const firstIn = Object.entries(a.inputs||{}).find(([,v])=>!isCtrl(v)&&fmtV(v));
  const preview = firstIn ? firstIn[0] : '';
  return '<div class="col-card-row">'
    +(hp?'<span class="sg-dot dot-pending"></span>':'')
    +'<span class="col-card-name">'+esc(a.name)+'</span>'
    +'</div>'
    +(preview?'<div class="col-card-preview">'+esc(preview)+'</div>':'');
}

function createColActCard(key){
  const a = byKey[key];
  if(!a||a.name==='assign') return;
  ensureColGroup(a.lifeline);
  const el = document.createElement('div');
  el.className = 'col-card col-act-'+a.kind+(a.kind==='human'?' col-pending':'');
  el.dataset.key = key;
  el.innerHTML = colActCardInner(a);
  el.onclick = function(){ selectColCard(key); };
  colActCards[key] = el;
  colPlace(el, a.lifeline, colNextRow(a.lifeline));
}

function updateColActCard(key){
  const el = colActCards[key]; if(!el) return;
  const a  = byKey[key]; if(!a) return;
  const req = a.reqId ? reqMap.get(a.reqId) : null;
  const hp  = req && !req.resolved && a.kind==='human';
  el.className = 'col-card col-act-'+a.kind+(hp?' col-pending':'')+(detailKey===key?' col-sel':'');
  el.innerHTML = colActCardInner(a);
  el.onclick = function(){ selectColCard(key); };
}

function createColMsgCard(e){
  if(isCtrlMsg(e)) return;
  const ll      = e.type==='send' ? e.from : e.to;
  const arrow   = e.type==='send' ? '→' : '←';
  const partner = e.type==='send' ? e.to : e.from;
  ensureColGroup(ll);
  const mk = msgKey(e);
  // Accumulate bindings from send + recv into shared byKey entry
  if(!byKey[mk]){
    byKey[mk] = {key:mk, kind:'msg', from:e.from, to:e.to, channel:e.channel||'', seq:e.seq, bindings:{}};
  }
  Object.assign(byKey[mk].bindings, e.bindings||{});
  const bindings = e.bindings||{};
  const vars = Object.keys(bindings).filter(function(k){
    return k!=='branch'&&k!=='loop'&&!isCtrl(bindings[k]);
  }).join(', ');
  const el = document.createElement('div');
  el.className = 'col-card col-msg col-msg-'+e.type;
  el.dataset.msgkey = mk;
  el.innerHTML = '<div class="col-card-row">'
    +'<span class="col-msg-arrow">'+esc(arrow)+'</span>'
    +'<span class="col-msg-partner">'+esc(partner)+'</span>'
    +'</div>'
    +(vars?'<div class="col-card-preview">'+esc(vars)+'</div>':'');
  el.onclick = function(){ selectColMsg(mk, el); };
  if(!msgCards[mk]) msgCards[mk] = {};
  msgCards[mk][e.type] = el;
  let row;
  if(e.type==='send'){
    row = colNextRow(ll);
    sendRowIdx[mk] = row;
  } else {
    row = Math.max(colNextRow(ll), sendRowIdx[mk]!==undefined ? sendRowIdx[mk] : 0);
  }
  colPlace(el, ll, row);
}

function handleDecision(e){
  if(!decisionsEnabled) return;
  const ll = e.lifeline;
  ensureColGroup(ll);
  const ctrlId = _ctrlSeq++;
  currentDecision[ll] = ctrlId;
  const decKey = 'dec:'+ctrlId;
  byKey[decKey] = {key:decKey, kind:'decision', lifeline:ll, value:e.value,
    formula:e.formula||'', condition:e.condition||'', decKind:e.kind||'if'};
  const sym  = e.value ? '⊤' : '⊥';
  const cond = e.formula||e.condition||'';
  const el   = document.createElement('div');
  el.className = 'col-decision';
  el.innerHTML = '<span class="col-dec-label">'+esc(e.kind||'if')+' '+sym+'</span>'
    +(cond?'<div class="col-dec-cond">'+esc(cond)+'</div>':'');
  el.onclick = function(){ openDetail(decKey); };
  decisionEls[decKey] = el;
  const decRow = Object.keys(colEls).reduce(function(m,k){ return Math.max(m,colNextRow(k)); }, colNextRow(ll));
  const y = rowToY(decRow);
  el.style.top = y+'px';
  colEls[ll].appendChild(el);
  colRowIdx[ll] = decRow+1;
  colSetScrollH(ll, y+ROW_H+COL_PAD);
  decisionRows.add(decRow);
  ctrlCards[ctrlId] = {row: decRow, decKey};
}

function handleCtrlRecv(e){
  const ctrlId = currentDecision[e.from];
  if(ctrlId===undefined) return;
  const sync = ctrlCards[ctrlId]; if(!sync) return;
  const ll = e.to;
  ensureColGroup(ll);
  const b    = e.bindings||{};
  const flag = b.branch==='true'||b.loop==='continue';
  const el   = document.createElement('div');
  el.className = 'col-ctrl-circle';
  el.textContent = flag ? '⊤' : '⊥';
  if(sync.decKey) el.onclick = function(){ openDetail(sync.decKey); };
  const y = rowToY(sync.row);
  el.style.top  = (y+5)+'px';
  el.style.left = 'calc(50% - 20px)';
  colEls[ll].appendChild(el);
  colRowIdx[ll] = Math.max(colRowIdx[ll]!==undefined?colRowIdx[ll]:0, sync.row+1);
  colSetScrollH(ll, y+ROW_H+COL_PAD);
}

function selectColCard(key){
  document.querySelectorAll('.col-msg.col-sel').forEach(function(el){ el.classList.remove('col-sel'); });
  openDetail(key);
}

function selectColMsg(mk, clickedEl){
  openDetail(mk);  // openDetail handles col-sel on both cards
  const pair = msgCards[mk]; if(!pair) return;
  const otherEl = clickedEl===pair.send ? pair.recv : pair.send;
  if(otherEl) alignCol(clickedEl, otherEl);
}

function alignCol(anchorEl, targetEl){
  const viewRect    = colView.getBoundingClientRect();
  const anchorOff   = anchorEl.getBoundingClientRect().top - viewRect.top;
  const newScrollTop = colView.scrollTop + targetEl.getBoundingClientRect().top - viewRect.top - anchorOff;
  colView.scrollTo({top: Math.max(0,newScrollTop), behavior:'smooth'});
}

colView.addEventListener('click', function(e){
  if(!e.target.closest('.col-card')){
    document.querySelectorAll('.col-msg.col-sel').forEach(function(el){ el.classList.remove('col-sel'); });
  }
});

// ── Event handlers ────────────────────────────────────────────────────────────
function handleInit(e){
  decisionsEnabled = e.show_decisions !== false;
  closeDetail();
  Object.keys(byKey).forEach(function(k){ delete byKey[k]; });
  Object.keys(groups).forEach(function(k){ delete groups[k]; });
  Object.keys(inboxCards).forEach(function(k){ delete inboxCards[k]; });
  lifelines.length = 0; reqMap.clear(); pending = 0;
  inboxList.innerHTML = '<p class="inbox-empty">No actions yet…</p>';
  updateInboxBadge();
  (e.lifelines||[]).forEach(function(ll){ ensureGroup(ll); });
  Object.keys(colEls).forEach(function(k){ delete colEls[k]; });
  Object.keys(colActCards).forEach(function(k){ delete colActCards[k]; });
  Object.keys(msgCards).forEach(function(k){ delete msgCards[k]; });
  Object.keys(colRowIdx).forEach(function(k){ delete colRowIdx[k]; });
  Object.keys(sendRowIdx).forEach(function(k){ delete sendRowIdx[k]; });
  Object.keys(currentDecision).forEach(function(k){ delete currentDecision[k]; });
  Object.keys(ctrlCards).forEach(function(k){ delete ctrlCards[k]; });
  Object.keys(decisionEls).forEach(function(k){ delete decisionEls[k]; });
  decisionRows.clear(); globalColH = 0; _ctrlSeq = 0;
  arrowsGroup.innerHTML = '';
  colView.innerHTML = '<p class="col-empty">Awaiting workflow…</p>';
  document.getElementById('btn-arrows').style.display = 'none';
  (e.lifelines||[]).forEach(function(ll){ ensureColGroup(ll); });
}

function handleRunStart(e){
  decisionRows.clear(); globalColH = 0;
  (e.lifelines||[]).forEach(function(ll){ ensureGroup(ll); ensureColGroup(ll); });
}

function handleActStart(e){
  const ll   = e.lifeline; ensureGroup(ll);
  const kind = e.action_kind||'pure', name = e.action||'—';
  if(name==='assign') return;
  const key  = ll+':'+e.seq;
  const now  = new Date();
  const time = now.getHours().toString().padStart(2,'0')+':'+now.getMinutes().toString().padStart(2,'0');
  byKey[key] = {key,lifeline:ll,name,kind,seq:e.seq,status:'pending',inputs:{},outputs:{},reqId:null,time};
  groups[ll].push(key);
  if(kind==='human') createInboxCard(key);
  createColActCard(key);
}

function handleAct(e){
  const key = e.lifeline+':'+e.seq;
  const a   = byKey[key]; if(!a) return;
  a.status = 'done'; a.inputs = e.inputs||{}; a.outputs = e.outputs||{};
  updateInboxCard(key);
  updateColActCard(key);
  if(detailKey===key) renderInspector(key, detailBody, closeDetail);
}

function handleHumanRequired(e){
  pending++; updateInboxBadge();
  reqMap.set(e.id, {lifeline:e.lifeline,kind:e.kind||'confirm',context:e.context??null,
    instruction:e.instruction??null,prefill:e.prefill??null,
    submit_label:e.submit_label??null,cancel_label:e.cancel_label??null,
    resolved:false,value:null});
  const llKeys = groups[e.lifeline]||[];
  let matchedKey = null;
  for(let i=llKeys.length-1; i>=0; i--){
    const k=llKeys[i], a=byKey[k];
    if(a&&a.kind==='human'&&a.status==='pending'&&!a.reqId){
      a.reqId = e.id; matchedKey = k; break;
    }
  }
  if(matchedKey){
    updateInboxCard(matchedKey);
    updateColActCard(matchedKey);
    const curReq = detailKey&&byKey[detailKey]&&byKey[detailKey].reqId
      ? reqMap.get(byKey[detailKey].reqId) : null;
    if(!curReq||curReq.resolved) openDetail(matchedKey);
  }
}

function handleHumanInput(e){
  const req = reqMap.get(e.id);
  if(req){ req.resolved=true; req.value=e.value; }
  pending = Math.max(0, pending-1); updateInboxBadge();
  Object.keys(byKey).forEach(function(k){
    if(byKey[k].reqId===e.id){ updateInboxCard(k); updateColActCard(k); }
  });
  if(detailKey&&byKey[detailKey]&&byKey[detailKey].reqId===e.id){
    renderInspector(detailKey, detailBody, closeDetail);
  }
  const entries = [...reqMap];
  for(let i=entries.length-1; i>=0; i--){
    const [id,r] = entries[i];
    if(!r.resolved){
      setTimeout(function(){
        Object.keys(byKey).forEach(function(k){ if(byKey[k].reqId===id) openDetail(k); });
      }, 350);
      return;
    }
  }
}

function handleSend(e){ createColMsgCard(e); }
function handleRecv(e){ if(isCtrlMsg(e)) handleCtrlRecv(e); else createColMsgCard(e); }

// ── Dispatcher ────────────────────────────────────────────────────────────────
function dispatch(e){
  if(!e||!e.type) return;
  switch(e.type){
    case 'init':                 handleInit(e); break;
    case 'run_start':            handleRunStart(e); break;
    case 'act_start':            handleActStart(e); break;
    case 'act':                  handleAct(e); break;
    case 'human_input_required': handleHumanRequired(e); break;
    case 'human_input':          handleHumanInput(e); break;
    case 'send':                 handleSend(e); break;
    case 'recv':                 handleRecv(e); break;
    case 'decision':             handleDecision(e); break;
  }
}

// ── SSE ───────────────────────────────────────────────────────────────────────
function connect(){
  if(evSrc){ evSrc.close(); evSrc=null; }
  const src = new EventSource('/events');
  evSrc = src;
  src.onmessage = function(ev){
    try{ dispatch(JSON.parse(ev.data)); }catch(err){ console.error(err,ev.data); }
  };
}
connect();

// ── Keyboard nav ──────────────────────────────────────────────────────────────
document.addEventListener('keydown', function(e){
  if(e.key==='Escape'){ closeDetail(); return; }
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT') return;
  if(e.key!=='ArrowUp'&&e.key!=='ArrowDown') return;
  e.preventDefault();
  const keys = Array.from(inboxList.querySelectorAll('.inbox-card')).map(function(el){ return el.dataset.key; });
  if(!keys.length) return;
  const idx  = keys.indexOf(detailKey);
  const next = e.key==='ArrowDown'
    ? keys[Math.min(idx+1, keys.length-1)]
    : keys[Math.max(idx-1, 0)];
  if(next && next!==detailKey) openDetail(next);
});
</script>
</body>
</html>
"""
