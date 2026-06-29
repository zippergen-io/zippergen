from zipperchat import WebTrace


def test_webtrace_hides_decisions_by_default():
    trace = WebTrace(["A", "B"], port=0)
    assert trace._init_event()["show_decisions"] is False


def test_webtrace_can_show_decisions_explicitly():
    trace = WebTrace(["A", "B"], port=0, show_decisions=True)
    assert trace._init_event()["show_decisions"] is True


def test_dashboard_hides_decisions_by_default():
    trace = WebTrace.dashboard(port=0)
    assert trace._init_event()["show_decisions"] is False


def test_dashboard_run_scopes_event_paths():
    trace = WebTrace.dashboard(port=0)
    trace.reset()
    run_trace = trace.start_run("alpha", ["A", "B"])
    run_path = run_trace.path

    run_trace({"type": "act", "lifeline": "A"})
    run_trace({"type": "level_push", "path": ["inner"], "lifelines": ["C"]})
    run_trace.done()

    history = trace._bus._history
    assert history[0] == {"type": "init", "dashboard": True, "show_decisions": False}
    assert history[1]["type"] == "run_start"
    assert history[1]["path"] == run_path
    assert history[2]["path"] == run_path
    assert history[3]["path"] == run_path + ["inner"]
    assert history[4] == {"type": "done", "path": run_path}
