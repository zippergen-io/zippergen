"""
Generate demo/index.html — a self-contained static ZipperChat demo.

Usage:
    python demo/build.py

Edit demo/scenario.py to change the email, LLM outputs, or human responses.
The generated demo/index.html can be served as a plain static file.
"""

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zipperchat.web import _HTML  # noqa: E402 — after sys.path patch


# ── Load scenario ─────────────────────────────────────────────────────────────

spec = importlib.util.spec_from_file_location("scenario", ROOT / "demo" / "scenario.py")
s = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(s)  # type: ignore[union-attr]


# ── Build event sequence ──────────────────────────────────────────────────────

LIFELINES = ["Mailbox", "Dispatcher", "Calendar", "Writer", "User"]

def make_events(s) -> list:
    """
    Return a list of [delay_ms, event_dict] pairs that replay the scheduling
    branch of command_center for the scenario email.

    human_input_required events are pause points — the replay waits for the
    user to submit the form before continuing.
    human_input events are NOT pre-recorded; they are dispatched by the demo JS.
    """
    E  = s.EMAIL
    AV = s.AVAILABILITY
    SC = s.SLOT_CHOICE
    R  = s.SCHEDULING_REPLY.replace(SC, "__SLOT__")
    RE = s.REPLY_EDIT.replace(SC, "__SLOT__")
    ED = s.EVENT_DETAILS_JSON
    ES = s.EVENT_SUMMARY
    TO = s.TODAY

    _seq  = [0]
    _mseq = [0]

    def seq():
        _seq[0] += 1
        return _seq[0]

    def ms():
        v = _mseq[0]
        _mseq[0] += 1
        return v

    ev = []  # [(delay_ms, event), ...]

    def e(delay, event):
        ev.append([delay, event])

    # ── Initialise ────────────────────────────────────────────────────────────
    e(0,   {"type": "init", "lifelines": LIFELINES, "show_decisions": False})
    e(200, {"type": "run_start", "name": "command_center", "lifelines": LIFELINES})

    # ── Mailbox polls inbox ───────────────────────────────────────────────────
    s1 = seq()
    e(400, {"type": "act_start", "lifeline": "Mailbox", "action": "pop_pending_email",
            "action_kind": "pure", "inputs": {}, "seq": s1})
    e(500, {"type": "act", "lifeline": "Mailbox", "action": "pop_pending_email",
            "action_kind": "pure", "inputs": {}, "outputs": {"email": E}, "seq": s1})

    ms0 = ms()
    e(150, {"type": "send", "from": "Mailbox", "to": "Dispatcher", "channel": "",
            "bindings": {"email": E}, "values": [E], "seq": ms0})
    e(100, {"type": "recv", "to": "Dispatcher", "from": "Mailbox", "channel": "",
            "bindings": {"email": E}, "seq": ms0})

    # ── Dispatcher classifies ─────────────────────────────────────────────────
    s2 = seq()
    e(200, {"type": "act_start", "lifeline": "Dispatcher", "action": "classify_email",
            "action_kind": "llm", "inputs": {"email": E}, "seq": s2})
    e(1300, {"type": "act", "lifeline": "Dispatcher", "action": "classify_email",
             "action_kind": "llm", "inputs": {"email": E},
             "outputs": {"route": s.CLASSIFY_ROUTE}, "seq": s2})

    s3 = seq()
    e(150, {"type": "act_start", "lifeline": "Dispatcher", "action": "normalize_route",
            "action_kind": "pure", "inputs": {"route": s.CLASSIFY_ROUTE}, "seq": s3})
    e(200, {"type": "act", "lifeline": "Dispatcher", "action": "normalize_route",
            "action_kind": "pure", "inputs": {"route": s.CLASSIFY_ROUTE},
            "outputs": {"route": "scheduling"}, "seq": s3})

    # ── Dispatcher → Calendar ─────────────────────────────────────────────────
    ms1 = ms()
    e(200, {"type": "send", "from": "Dispatcher", "to": "Calendar", "channel": "",
            "bindings": {"email": E}, "values": [E], "seq": ms1})
    e(100, {"type": "recv", "to": "Calendar", "from": "Dispatcher", "channel": "",
            "bindings": {"email": E}, "seq": ms1})

    s4 = seq()
    e(200, {"type": "act_start", "lifeline": "Calendar",
            "action": "classify_scheduling_request",
            "action_kind": "llm", "inputs": {"email": E}, "seq": s4})
    e(1100, {"type": "act", "lifeline": "Calendar",
             "action": "classify_scheduling_request",
             "action_kind": "llm", "inputs": {"email": E},
             "outputs": {"sched_kind": s.SCHED_KIND}, "seq": s4})

    s5 = seq()
    e(150, {"type": "act_start", "lifeline": "Calendar", "action": "normalize_route",
            "action_kind": "pure", "inputs": {"sched_kind": s.SCHED_KIND}, "seq": s5})
    e(200, {"type": "act", "lifeline": "Calendar", "action": "normalize_route",
            "action_kind": "pure", "inputs": {"sched_kind": s.SCHED_KIND},
            "outputs": {"sched_kind": "propose_slots"}, "seq": s5})

    # ── Calendar proposes slots ───────────────────────────────────────────────
    s6 = seq()
    e(200, {"type": "act_start", "lifeline": "Calendar", "action": "propose_available_slots",
            "action_kind": "pure", "inputs": {"email": E}, "seq": s6})
    e(400, {"type": "act", "lifeline": "Calendar", "action": "propose_available_slots",
            "action_kind": "pure", "inputs": {"email": E},
            "outputs": {"availability": AV}, "seq": s6})

    ms2 = ms()
    e(150, {"type": "send", "from": "Calendar", "to": "User", "channel": "",
            "bindings": {"email": E, "availability": AV}, "values": [E, AV], "seq": ms2})
    e(100, {"type": "recv", "to": "User", "from": "Calendar", "channel": "",
            "bindings": {"email": E, "availability": AV}, "seq": ms2})

    # ── Human: choose slot ────────────────────────────────────────────────────
    s7 = seq()
    e(200, {"type": "act_start", "lifeline": "User", "action": "choose_from_proposed_slots",
            "action_kind": "human", "inputs": {"email": E, "availability": AV}, "seq": s7})
    e(300, {"type": "human_input_required", "id": "req-slot",
            "lifeline": "User", "kind": "edit",
            "context": E + "\n\nAvailable slots:\n" + AV.replace(" or ", "\n"),
            "instruction": "Keep the slot that applies, delete the other",
            "prefill": AV.replace(" or ", "\n"),
            "submit_label": "Select →", "cancel_label": "Decline"})
    # Replay resumes after the user submits (or after _demoReqResolve fires)
    SLOT = "__SLOT__"  # substituted at runtime with the user's actual choice
    e(100, {"type": "act", "lifeline": "User", "action": "choose_from_proposed_slots",
            "action_kind": "human", "inputs": {"email": E, "availability": AV},
            "outputs": {"choice": SLOT}, "seq": s7})

    # ── User → Writer ─────────────────────────────────────────────────────────
    ms3 = ms()
    e(200, {"type": "send", "from": "User", "to": "Writer", "channel": "",
            "bindings": {"email": E, "choice": SLOT}, "values": [E, SLOT], "seq": ms3})
    e(100, {"type": "recv", "to": "Writer", "from": "User", "channel": "",
            "bindings": {"email": E, "choice": SLOT}, "seq": ms3})

    s8 = seq()
    e(200, {"type": "act_start", "lifeline": "Writer", "action": "write_scheduling_reply",
            "action_kind": "llm", "inputs": {"email": E, "choice": SLOT}, "seq": s8})
    e(1200, {"type": "act", "lifeline": "Writer", "action": "write_scheduling_reply",
             "action_kind": "llm", "inputs": {"email": E, "choice": SLOT},
             "outputs": {"reply": R}, "seq": s8})

    # ── approve_email_reply ───────────────────────────────────────────────────
    ms4 = ms()
    e(200, {"type": "send", "from": "Writer", "to": "User", "channel": "",
            "bindings": {"email": E, "reply": R}, "values": [E, R], "seq": ms4})
    e(100, {"type": "recv", "to": "User", "from": "Writer", "channel": "",
            "bindings": {"email": E, "reply": R}, "seq": ms4})

    s9 = seq()
    e(200, {"type": "act_start", "lifeline": "User", "action": "approve_or_edit",
            "action_kind": "human", "inputs": {"email": E, "reply": R}, "seq": s9})
    e(300, {"type": "human_input_required", "id": "req-reply",
            "lifeline": "User", "kind": "edit",
            "context": E,
            "instruction": "Edit the proposed reply or submit as-is",
            "prefill": RE,
            "submit_label": "Save draft →", "cancel_label": "Skip"})
    e(100, {"type": "act", "lifeline": "User", "action": "approve_or_edit",
            "action_kind": "human", "inputs": {"email": E, "reply": R},
            "outputs": {"edit": RE}, "seq": s9})

    # ── User → Mailbox (create draft) ────────────────────────────────────────
    ms5 = ms()
    e(200, {"type": "send", "from": "User", "to": "Mailbox", "channel": "",
            "bindings": {"email": E, "edit": RE}, "values": [E, RE], "seq": ms5})
    e(100, {"type": "recv", "to": "Mailbox", "from": "User", "channel": "",
            "bindings": {"email": E, "edit": RE}, "seq": ms5})

    s10 = seq()
    e(200, {"type": "act_start", "lifeline": "Mailbox", "action": "create_draft",
            "action_kind": "pure", "inputs": {"email": E, "reply": RE}, "seq": s10})
    e(400, {"type": "act", "lifeline": "Mailbox", "action": "create_draft",
            "action_kind": "pure", "inputs": {"email": E, "reply": RE},
            "outputs": {"mail_status": "draft_created"}, "seq": s10})

    # ── User → Calendar (create event) ───────────────────────────────────────
    ms6 = ms()
    e(200, {"type": "send", "from": "User", "to": "Calendar", "channel": "",
            "bindings": {"email": E, "choice": SLOT, "reply": RE},
            "values": [E, SLOT, RE], "seq": ms6})
    e(100, {"type": "recv", "to": "Calendar", "from": "User", "channel": "",
            "bindings": {"email": E, "choice": SLOT, "reply": RE}, "seq": ms6})

    s11 = seq()
    e(200, {"type": "act_start", "lifeline": "Calendar",
            "action": "remember_scheduling_reply",
            "action_kind": "pure",
            "inputs": {"email": E, "reply": RE, "sched_context": ""}, "seq": s11})
    e(300, {"type": "act", "lifeline": "Calendar",
            "action": "remember_scheduling_reply",
            "action_kind": "pure",
            "inputs": {"email": E, "reply": RE, "sched_context": ""},
            "outputs": {"sched_context": RE[:80]}, "seq": s11})

    s12 = seq()
    e(200, {"type": "act_start", "lifeline": "Calendar", "action": "todays_date",
            "action_kind": "pure", "inputs": {}, "seq": s12})
    e(200, {"type": "act", "lifeline": "Calendar", "action": "todays_date",
            "action_kind": "pure", "inputs": {}, "outputs": {"today": TO}, "seq": s12})

    s13 = seq()
    e(200, {"type": "act_start", "lifeline": "Calendar", "action": "extract_event_details",
            "action_kind": "llm", "inputs": {"today": TO, "email": E, "choice": SLOT},
            "seq": s13})
    e(1000, {"type": "act", "lifeline": "Calendar", "action": "extract_event_details",
             "action_kind": "llm", "inputs": {"today": TO, "email": E, "choice": SLOT},
             "outputs": {"event_details": ED}, "seq": s13})

    s14 = seq()
    e(200, {"type": "act_start", "lifeline": "Calendar", "action": "create_scheduled_event",
            "action_kind": "pure", "inputs": {"event_details": ED}, "seq": s14})
    e(300, {"type": "act", "lifeline": "Calendar", "action": "create_scheduled_event",
            "action_kind": "pure", "inputs": {"event_details": ED},
            "outputs": {"sched_event": "skipped (mock mode)"}, "seq": s14})

    ES_TMPL = "ZipperGen Research Call — __SLOT__"
    s15 = seq()
    e(200, {"type": "act_start", "lifeline": "Calendar",
            "action": "format_event_confirmation",
            "action_kind": "pure", "inputs": {"event_details": ED}, "seq": s15})
    e(200, {"type": "act", "lifeline": "Calendar",
            "action": "format_event_confirmation",
            "action_kind": "pure", "inputs": {"event_details": ED},
            "outputs": {"event_summary": ES_TMPL}, "seq": s15})

    ms7 = ms()
    e(200, {"type": "send", "from": "Calendar", "to": "User", "channel": "",
            "bindings": {"event_summary": ES_TMPL}, "values": [ES_TMPL], "seq": ms7})
    e(100, {"type": "recv", "to": "User", "from": "Calendar", "channel": "",
            "bindings": {"event_summary": ES_TMPL}, "seq": ms7})

    # ── Human: acknowledge event ──────────────────────────────────────────────
    s16 = seq()
    e(200, {"type": "act_start", "lifeline": "User",
            "action": "acknowledge_event_created",
            "action_kind": "human", "inputs": {"event_summary": ES_TMPL}, "seq": s16})
    e(300, {"type": "human_input_required", "id": "req-ack",
            "lifeline": "User", "kind": "ack",
            "context": ES_TMPL, "instruction": "Calendar event created.",
            "prefill": None, "submit_label": "Noted", "cancel_label": None})
    e(100, {"type": "act", "lifeline": "User",
            "action": "acknowledge_event_created",
            "action_kind": "human", "inputs": {"event_summary": ES_TMPL},
            "outputs": {"ack": "true"}, "seq": s16})

    return ev


# ── HTML transformation ───────────────────────────────────────────────────────

def build_demo_html(events: list, logo_svg: str) -> str:
    html = _HTML

    # Inline the logo SVG and link it back to the main site.
    html = html.replace(
        '<img src="/assets/zippergen-lockup-ink.svg" alt="ZipperGen" class="hdr-logo">',
        '<a class="hdr-logo-link" href="https://zippergen.io" '
        f'aria-label="ZipperGen home">{logo_svg}</a>',
    )

    # Inject CSS for the replay button (same look as arrows button)
    extra_css = (
        "\n#btn-replay {\n"
        "  background: none; border: 1px solid transparent;\n"
        "  font-size: 13px; font-weight: 500; color: var(--text-faint);\n"
        "  padding: 4px 10px; border-radius: 5px; cursor: pointer;\n"
        "}\n"
        "#btn-replay:hover { color: var(--text-soft); }\n"
        ".hdr-logo-link { display: inline-flex; align-items: center; text-decoration: none; }\n"
    )
    html = html.replace("</style>", extra_css + "</style>", 1)

    # Add Replay button next to the arrows button (no inline style)
    html = html.replace(
        '<button id="btn-arrows"',
        '<button id="btn-replay" onclick="_demoRestart()">Replay</button>'
        '<button id="btn-arrows"',
    )

    # Replace the SSE connect() call with the demo replay engine
    ll_json = json.dumps(LIFELINES)
    events_json = json.dumps(events, ensure_ascii=False)
    sc_json = json.dumps(s.SLOT_CHOICE, ensure_ascii=False)
    replay_js = f"""
// ── Demo replay engine ────────────────────────────────────────────────────────
const _DEMO_EVENTS = {events_json};
const _SC_DEFAULT = {sc_json};
let _chosenSlot = _SC_DEFAULT;

let _demoIdx = 0;
let _demoPaused = false;

// Show arrows by default
showArrows = true;
document.getElementById('btn-arrows').classList.add('arr-on');

function _subSlot(obj) {{
  if (typeof obj === 'string') return obj.includes('__SLOT__') ? obj.replaceAll('__SLOT__', _chosenSlot) : obj;
  if (Array.isArray(obj)) return obj.map(_subSlot);
  if (obj && typeof obj === 'object') {{
    const r = {{}};
    for (const k in obj) r[k] = _subSlot(obj[k]);
    return r;
  }}
  return obj;
}}

function _demoStep() {{
  if (_demoIdx >= _DEMO_EVENTS.length) return;
  const pair = _DEMO_EVENTS[_demoIdx++];
  const delay = pair[0];
  const ev = _subSlot(pair[1]);
  setTimeout(function () {{
    dispatch(ev);
    if (ev.type === 'human_input_required') {{
      _demoPaused = true;
    }} else {{
      _demoStep();
    }}
  }}, delay);
}}

// Override doSubmit so form buttons work without a server
window.doSubmit = function (req_id, val) {{
  val = val.trim();
  if (req_id === 'req-slot') _chosenSlot = val || _chosenSlot;
  dispatch({{type: 'human_input', id: req_id, value: val}});
  _demoPaused = false;
  _demoStep();
}};

function _demoRestart() {{
  _demoIdx = 0;
  _demoPaused = false;
  _chosenSlot = _SC_DEFAULT;
  dispatch({{type: 'init', lifelines: {ll_json}, show_decisions: false}});
  showArrows = true;
  document.getElementById('btn-arrows').classList.add('arr-on');
  _demoStep();
}}

_demoStep();
"""

    html = html.replace("connect();", replay_js)
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logo_path = ROOT / "src" / "zipperchat" / "assets" / "zippergen-lockup-ink.svg"
    logo_svg = logo_path.read_text(encoding="utf-8")
    logo_svg = logo_svg.removeprefix('<?xml version="1.0" encoding="UTF-8"?>\n')
    logo_svg = logo_svg.replace(
        "<svg ",
        '<svg class="hdr-logo" style="height:36px;width:auto" ',
        1,
    )

    events = make_events(s)
    html = build_demo_html(events, logo_svg)

    out = ROOT / "demo" / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Written {out}  ({len(html):,} bytes, {len(events)} events)")
