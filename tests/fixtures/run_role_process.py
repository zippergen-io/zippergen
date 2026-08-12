"""Test-only subprocess entry point for one durable projected role."""

from __future__ import annotations

import argparse
import json

from zippergen.projection import project
from zippergen.role_runner import run_role
from zippergen.store import open_store
from zippergen.workflow_io import _workflow_lifelines, load_workflow_spec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--input", action="append", default=[])
    args = parser.parse_args()

    inputs: dict[str, object] = {}
    for pair in args.input:
        name, separator, raw = pair.partition("=")
        if not name or not separator:
            parser.error(f"invalid --input {pair!r}; expected name=value")
        try:
            inputs[name] = json.loads(raw)
        except json.JSONDecodeError:
            inputs[name] = raw

    workflow, _module = load_workflow_spec(args.workflow)
    lifelines = {item.name: item for item in _workflow_lifelines(workflow)}
    if args.role not in lifelines:
        parser.error(f"unknown role {args.role!r}")
    connection = open_store(args.store)
    # The starting inputs are only used the first time; a restarted role reads
    # its own committed environment instead.
    result = run_role(
        connection,
        args.role,
        project(workflow, lifelines[args.role]),
        inputs,
        workflow.ns,
    )
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
