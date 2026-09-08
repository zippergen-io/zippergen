#!/usr/bin/env python3
"""Measure the time and message data used by the ZipperGen CPL monitor."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_ZIPPERGEN_SRC = REPOSITORY_ROOT / "src"
RESULTS_PATHSPEC = "experiments/icfem-2026-cpl/results/**"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zippergen-src",
        type=Path,
        default=DEFAULT_ZIPPERGEN_SRC,
        help="path containing the zippergen package (default: this checkout's src)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="operations per repetition (default: 300)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=9,
        help="timing repetitions per operation and configuration (default: 9)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="untimed warmup operations (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "results" / "latest",
        help="output stem for .csv and .md files",
    )
    parser.add_argument(
        "--cpu-model",
        default=None,
        help="processor name to record when automatic detection is unavailable",
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 2 or args.warmup < 0:
        parser.error(
            "iterations must be positive; repeats must be at least 2; "
            "warmup must be nonnegative"
        )
    return args


ARGS = parse_args()
ZIPPERGEN_SRC = ARGS.zippergen_src.resolve()
if not (ZIPPERGEN_SRC / "zippergen" / "monitor.py").is_file():
    raise SystemExit(f"ZipperGen monitor not found below {ZIPPERGEN_SRC}")
sys.path.insert(0, str(ZIPPERGEN_SRC))

from zippergen.formula import At, Here, AnyFormula, atom, since, subformulas  # noqa: E402
from zippergen.monitor import MonitorState  # noqa: E402
from zippergen.store import _encode_causal_stamp  # noqa: E402


def make_subformulas(count: int, fields: int) -> list[AnyFormula]:
    """Build a conjunction with exactly count subformulas.

    Each atom compares a field on L0 with a distinct integer constant. A
    conjunction of n atoms has 2n-1 subformulas.
    """

    if count % 2 != 1:
        raise ValueError("the conjunction workload requires an odd node count")
    atom_count = (count + 1) // 2
    if atom_count < fields:
        raise ValueError("the benchmark needs at least one atom per field")
    leaves = [
        getattr(At["L0"], f"field_{index % fields}") == index
        for index in range(atom_count)
    ]
    guard = leaves[0]
    for leaf in leaves[1:]:
        guard = guard & leaf
    nodes = subformulas(guard)
    if len(nodes) != count:
        raise RuntimeError(
            f"constructed {len(nodes)} subformulas instead of {count}"
        )
    return nodes


def code_review_subformulas() -> list[AnyFormula]:
    test_not_failed = atom(
        lambda env: env.get("status") != "failed",
        src="status != failed",
        version="test-not-failed-v1",
    )
    not_pending = atom(
        lambda env: env.get("status") != "pending",
        src="status != pending",
        version="not-pending-v1",
    )
    test_passed = atom(
        lambda env: env.get("status") == "passed",
        src="status == passed",
        version="test-passed-v1",
    )
    security_not_critical = atom(
        lambda env: env.get("status") != "critical",
        src="status != critical",
        version="security-not-critical-v1",
    )
    security_cleared = atom(
        lambda env: env.get("status") == "cleared",
        src="status == cleared",
        version="security-cleared-v1",
    )
    guard = (
        (At["TestRunner"].candidate == Here.candidate)
        & (At["Security"].candidate == Here.candidate)
        & At["TestRunner"](
            since(test_not_failed & not_pending, test_passed)
        )
        & At["Security"](
            since(security_not_critical & not_pending, security_cleared)
        )
    )
    return subformulas(guard)


def make_env(fields: int) -> dict[str, object]:
    return {f"field_{index}": index for index in range(fields)}


def populate_full_state(
    monitor: MonitorState,
    formulas: list[AnyFormula],
    env: dict[str, object],
) -> None:
    """Fill every monitor table entry for a fixed-size update workload."""
    values = {id(formula): (index % 2 == 0) for index, formula in enumerate(formulas)}
    for index, lifeline in enumerate(monitor.lifelines, start=1):
        monitor.vc[lifeline] = index
        monitor.view[lifeline] = dict(values)
        monitor.field_view[lifeline] = dict(env)


def time_operation(operation: Callable[[], object]) -> tuple[float, float, float]:
    for _ in range(ARGS.warmup):
        operation()

    samples: list[float] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(ARGS.repeats):
            start = time.perf_counter_ns()
            for _ in range(ARGS.iterations):
                operation()
            elapsed = time.perf_counter_ns() - start
            samples.append(elapsed / ARGS.iterations / 1_000.0)
    finally:
        if was_enabled:
            gc.enable()

    median = statistics.median(samples)
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    iqr = quartiles[2] - quartiles[0]
    spread = max(samples) - min(samples)
    return median, iqr, spread


def benchmark_case(
    series: str,
    parameter: int | str,
    lifeline_count: int,
    subformula_count: int,
    field_count: int,
) -> dict[str, object]:
    if series == "running_example":
        lifelines = ["Orchestrator", "TestRunner", "Security", "Committer"]
        formulas = code_review_subformulas()
        env = {"candidate": "patch-B", "status": "checking"}
        field_count = 1
    else:
        lifelines = [f"L{index}" for index in range(lifeline_count)]
        formulas = make_subformulas(subformula_count, field_count)
        env = make_env(field_count)
    owner = lifelines[-1]

    local_monitor = MonitorState(owner, lifelines, formulas)
    populate_full_state(local_monitor, formulas, env)

    act_us, act_iqr_us, act_spread_us = time_operation(
        lambda: local_monitor.on_event("act", env)
    )

    send_monitor = MonitorState(owner, lifelines, formulas)
    populate_full_state(send_monitor, formulas, env)

    def send_and_snapshot() -> tuple[dict, dict, dict]:
        send_monitor.on_event("send", env)
        return (
            send_monitor.snapshot_vc(),
            send_monitor.snapshot_view(),
            send_monitor.snapshot_field_view(),
        )

    send_us, send_iqr_us, send_spread_us = time_operation(send_and_snapshot)

    receive_monitor = MonitorState(owner, lifelines, formulas)
    populate_full_state(receive_monitor, formulas, env)
    incoming_vc = dict(receive_monitor.vc)
    incoming_view = {
        lifeline: {
            index: (index % 2 == 0)
            for index in range(len(formulas))
        }
        for lifeline in lifelines
    }
    incoming_fields = {lifeline: dict(env) for lifeline in lifelines}

    def receive_new_frontier() -> None:
        for lifeline in lifelines[:-1]:
            incoming_vc[lifeline] += 1
        receive_monitor.on_event(
            "recv",
            env,
            recv_vc=incoming_vc,
            recv_view=incoming_view,
            recv_field_view=incoming_fields,
        )

    receive_us, receive_iqr_us, receive_spread_us = time_operation(receive_new_frontier)

    metadata_monitor = MonitorState(owner, lifelines, formulas)
    populate_full_state(metadata_monitor, formulas, env)
    stamp = _encode_causal_stamp(
        metadata_monitor.snapshot_vc(),
        metadata_monitor.snapshot_view(),
        metadata_monitor.snapshot_field_view(),
    )
    if stamp is None:
        raise RuntimeError("the populated monitor produced no message metadata")

    return {
        "series": series,
        "parameter": parameter,
        "lifelines": lifeline_count,
        "subformulas": len(formulas),
        "variables": field_count,
        "iterations": ARGS.iterations,
        "repeats": ARGS.repeats,
        "local_action_us": round(act_us, 3),
        "local_action_iqr_us": round(act_iqr_us, 3),
        "local_action_range_us": round(act_spread_us, 3),
        "send_us": round(send_us, 3),
        "send_iqr_us": round(send_iqr_us, 3),
        "send_range_us": round(send_spread_us, 3),
        "receive_us": round(receive_us, 3),
        "receive_iqr_us": round(receive_iqr_us, 3),
        "receive_range_us": round(receive_spread_us, 3),
        "message_metadata_bytes": len(stamp.encode("utf-8")),
    }


def cases() -> Iterable[tuple[str, int | str, int, int, int]]:
    yield "running_example", "guard", 4, 16, 1
    # Common configuration: 8 lifelines, 31 subformulas, 4 fields.
    for lifelines in (2, 4, 8, 16, 32):
        yield "lifelines", lifelines, lifelines, 31, 4
    for formulas in (7, 15, 63, 127):
        yield "subformulas", formulas, 8, formulas, 4
    for variables in (1, 4, 16, 64, 128):
        yield "variables", variables, 8, 255, variables


def git_revision(directory: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_diff_sha256(
    directory: Path,
    *,
    exclude: tuple[str, ...] = (),
) -> str | None:
    try:
        command = [
            "git", "-C", str(directory), "diff", "--binary", "HEAD", "--", ".",
        ]
        command.extend(f":(exclude){path}" for path in exclude)
        diff = subprocess.run(
            command,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(diff).hexdigest() if diff else None


def git_worktree_status(
    directory: Path,
    *,
    exclude: tuple[str, ...] = (),
) -> dict[str, str | None]:
    """Distinguish a clean tree from a failed check and include untracked paths."""
    command = [
        "git", "-C", str(directory), "status", "--porcelain=v1",
        "--untracked-files=all", "--", ".",
    ]
    command.extend(f":(exclude){path}" for path in exclude)
    try:
        status = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except OSError as error:
        return {"porcelain": None, "error": str(error)}
    except subprocess.CalledProcessError as error:
        return {"porcelain": None, "error": error.stderr.strip() or str(error)}
    return {"porcelain": status, "error": None}


def cpu_model() -> str:
    if ARGS.cpu_model:
        return str(ARGS.cpu_model)
    try:
        model = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        model = ""
    return model or platform.processor()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "series",
        "parameter",
        "lifelines",
        "subformulas",
        "variables",
        "local_action_us",
        "send_us",
        "receive_us",
        "message_metadata_bytes",
    ]
    labels = {
        "series": "series",
        "parameter": "value",
        "lifelines": "lifelines",
        "subformulas": "subformulas",
        "variables": "variables",
        "local_action_us": "local action (us)",
        "send_us": "send (us)",
        "receive_us": "receive (us)",
        "message_metadata_bytes": "message metadata (bytes)",
    }
    lines = [
        "# CPL monitor microbenchmark results",
        "",
        f"Median of {ARGS.repeats} repetitions, {ARGS.iterations} operations each.",
        "",
        "| " + " | ".join(labels[column] for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_environment() -> dict[str, object]:
    """Record the source state before the benchmark changes its result files."""
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_model(),
        "cpu_count": os.cpu_count(),
        "zippergen_source": str(ZIPPERGEN_SRC),
        "zippergen_revision": git_revision(ZIPPERGEN_SRC.parent),
        "zippergen_diff_sha256": git_diff_sha256(
            ZIPPERGEN_SRC.parent,
            exclude=(RESULTS_PATHSPEC,),
        ),
        "zippergen_worktree": git_worktree_status(
            ZIPPERGEN_SRC.parent,
            exclude=(RESULTS_PATHSPEC,),
        ),
        "benchmark_sha256": file_sha256(Path(__file__)),
        "iterations": ARGS.iterations,
        "repeats": ARGS.repeats,
        "warmup": ARGS.warmup,
    }


def write_environment(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    output_stem = ARGS.output.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    environment = collect_environment()

    rows = []
    for case in cases():
        row = benchmark_case(*case)
        rows.append(row)
        print(
            f"{row['series']:11s} {row['parameter']:>3}: "
            f"local={row['local_action_us']:>8} us, "
            f"send={row['send_us']:>8} us, "
            f"recv={row['receive_us']:>8} us, "
            f"metadata={row['message_metadata_bytes']} bytes",
            flush=True,
        )

    csv_path = output_stem.with_suffix(".csv")
    markdown_path = output_stem.with_suffix(".md")
    environment_path = output_stem.parent / "environment.json"
    write_csv(csv_path, rows)
    write_markdown(markdown_path, rows)
    write_environment(environment_path, environment)
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {environment_path}")


if __name__ == "__main__":
    main()
