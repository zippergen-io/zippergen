"""Shared runtime exceptions whose meaning crosses executor boundaries."""

from __future__ import annotations

__all__ = ["WorkflowCancelled"]


class WorkflowCancelled(RuntimeError):
    """Normal termination requested by the run or deployment supervisor."""
