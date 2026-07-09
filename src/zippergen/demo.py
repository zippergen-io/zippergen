"""Small helper for running example workflows with minimal boilerplate."""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["run_demo"]


def run_demo(
    *,
    workflow,
    inputs: dict[str, object],
    llm: str | Mapping[str, str] | None = None,
    llms: str | Mapping[str, str] | None = None,
    ui: bool = True,
    timeout: float = 60.0,
    mock_delay: tuple[float, float] = (1.0, 2.0),
    llm_idle_timeout: float | None = None,
    show_decisions: bool = False,
):
    """
    Run an example workflow with a minimal user-facing API.

    Parameters
    ----------
    workflow
        The workflow object to execute.
    inputs
        Keyword arguments passed to the workflow.
    llm
        Either:
        - ``None`` or ``"mock"`` for the built-in mock backend
        - a compact spec like ``"openai:gpt-4o"`` or ``"ollama:qwen2.5:7b"``
        - a mapping ``lifeline_name -> spec`` for per-lifeline routing
    llms
        Backward-compatible alias for ``llm``.
    ui
        If true, run the browser UI and keep replaying via the browser button.
        If false, run once with console output only.
    timeout
        Per-thread runtime timeout in seconds.
    mock_delay
        Delay range used by the mock backend.
    """
    workflow.configure(
        llm=llm,
        llms=llms,
        ui=ui,
        timeout=timeout,
        mock_delay=mock_delay,
        llm_idle_timeout=llm_idle_timeout,
        show_decisions=show_decisions,
    )
    result = workflow(**inputs)
    print(f"\nResult → {result}")
    return result
