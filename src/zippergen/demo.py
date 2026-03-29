"""Small helper for running example workflows with minimal boilerplate."""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["run_demo"]


def run_demo(
    *,
    workflow,
    inputs: dict[str, object],
    llms: str | Mapping[str, str] | None = None,
    ui: bool = True,
    timeout: float = 60.0,
    mock_delay: tuple[float, float] = (1.0, 2.0),
):
    """
    Run an example workflow with a minimal user-facing API.

    Parameters
    ----------
    workflow
        The workflow object to execute.
    inputs
        Keyword arguments passed to the workflow.
    llms
        Either:
        - ``None`` or ``"mock"`` for the built-in mock backend
        - a provider name like ``"openai"`` or ``"mistral"`` for all lifelines
        - a mapping ``lifeline_name -> provider`` for per-lifeline routing
    ui
        If true, run the browser UI and keep replaying via the browser button.
        If false, run once with console output only.
    timeout
        Per-thread runtime timeout in seconds.
    mock_delay
        Delay range used by the mock backend.
    """
    workflow.configure(
        llms=llms,
        ui=ui,
        timeout=timeout,
        mock_delay=mock_delay,
    )
    result = workflow(**inputs)
    print(f"\nResult → {result}")
    return result
