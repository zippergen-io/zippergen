"""Run the arithmetic planner experiment across several planner models.

The experiment checks whether a model can produce a ZipperGen workflow that is
accepted, projected, executed, and numerically correct for two arithmetic
expressions. Generated workflows are saved for manual inspection.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.arithmetic_planner import arithmetic_planner
from zippergen.backends import make_openai_backend


EXPRESSIONS = [
    {
        "case": "x=2",
        "expression": "(2 - 4) * (2 + 3) + (3 / (3 - 2))",
        "expected": -7.0,
    },
    {
        "case": "x=3",
        "expression": "(2 - 4) * (2 + 3) + (3 / (3 - 3))",
        "expected": 0.0,
    },
]


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    label: str
    model: str
    base_url: str
    api_key_env: str | None = None
    api_key_default: str | None = None


MODELS = [
    ModelConfig(
        label="gpt-4o-mini",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    ModelConfig(
        label="gpt-4.1-mini",
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    ModelConfig(
        label="gpt-4o",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    ModelConfig(
        label="gpt-4.1",
        model="gpt-4.1",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    ModelConfig(
        label="qwen2.5:7b-local",
        model="qwen2.5:7b",
        base_url="http://127.0.0.1:11434/v1",
        api_key_default="ollama",
    ),
    ModelConfig(
        label="qwen2.5:14b-local",
        model="qwen2.5:14b",
        base_url="http://127.0.0.1:11434/v1",
        api_key_default="ollama",
    ),
]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text).strip("-")


def _local_server_reachable(base_url: str, timeout: float = 2.0) -> bool:
    if not base_url.startswith("http://127.0.0.1:") and not base_url.startswith("http://localhost:"):
        return True
    request = urllib.request.Request(base_url.rstrip("/") + "/models", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except (OSError, urllib.error.URLError):
        return False


def _extract_workflow(output: str) -> tuple[str | None, int | None]:
    marker = "=" * 60
    attempts = None
    header = re.search(r"GENERATED WORKFLOW\s+\((\d+) attempt", output)
    if header:
        attempts = int(header.group(1))
    parts = output.split(marker)
    if len(parts) >= 4 and "GENERATED WORKFLOW" in parts[1]:
        return parts[2].strip(), attempts
    return None, attempts


def _make_backend(config: ModelConfig, *, temperature: float, max_tokens: int, timeout: float):
    api_key = config.api_key_default
    if config.api_key_env is not None:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"{config.api_key_env} is not set")
    if not _local_server_reachable(config.base_url):
        raise RuntimeError(f"server is not reachable at {config.base_url}")
    return make_openai_backend(
        api_key=api_key or "",
        model=config.model,
        base_url=config.base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def _run_one(
    config: ModelConfig,
    case: dict,
    run_index: int,
    out_dir: Path,
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> dict:
    row = {
        "model": config.label,
        "model_id": config.model,
        "case": case["case"],
        "expression": case["expression"],
        "expected": case["expected"],
        "run": run_index,
        "parse_at_1": False,
        "parse_at_k": False,
        "attempts": None,
        "valid": False,
        "executed": False,
        "correct": False,
        "result": None,
        "elapsed_ms": None,
        "workflow_file": None,
        "log_file": None,
        "error": None,
    }

    try:
        backend = _make_backend(config, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    except Exception as exc:
        row["error"] = str(exc)
        return row

    arithmetic_planner.configure(backend=backend, ui=False, timeout=timeout)

    buffer = io.StringIO()
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buffer):
            result = arithmetic_planner(expression=case["expression"])
        row["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        row["result"] = result
        row["executed"] = True
        row["correct"] = math.isclose(float(result), float(case["expected"]), rel_tol=0.0, abs_tol=1e-9)
    except Exception as exc:
        row["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        row["error"] = str(exc)

    output = buffer.getvalue()
    workflow, attempts = _extract_workflow(output)
    failed_attempts = len(re.findall(r"\[planner\] attempt \d+ failed:", output))
    if attempts is None and failed_attempts:
        attempts = failed_attempts
    row["attempts"] = attempts
    row["parse_at_k"] = workflow is not None
    row["parse_at_1"] = workflow is not None and attempts == 1
    row["valid"] = workflow is not None

    stem = f"{_slug(config.label)}_{case['case']}_run{run_index}"
    if output:
        log_path = out_dir / "logs" / f"{stem}.txt"
        log_path.write_text(output, encoding="utf-8")
        row["log_file"] = str(log_path)
    if workflow:
        workflow_path = out_dir / "workflows" / f"{stem}.py"
        workflow_path.write_text(workflow + "\n", encoding="utf-8")
        row["workflow_file"] = str(workflow_path)

    return row


def _print_summary(rows: list[dict]) -> None:
    print("| Model | Case | Parse@1 | Parse@k | Attempts | Valid | Executed | Correct | Result | Error |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        error_lines = (row["error"] or "").splitlines()
        error = error_lines[0] if error_lines else ""
        if len(error) > 70:
            error = error[:67] + "..."
        print(
            f"| {row['model']} | {row['case']} | "
            f"{int(row['parse_at_1'])} | {int(row['parse_at_k'])} | "
            f"{row['attempts'] if row['attempts'] is not None else '--'} | "
            f"{int(row['valid'])} | {int(row['executed'])} | {int(row['correct'])} | "
            f"{row['result'] if row['result'] is not None else '--'} | {error} |"
        )


def main() -> None:
    _load_env_file(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="runs per model/expression")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[config.label for config in MODELS],
        default=None,
        help="model labels to run; defaults to all configured models",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory; defaults to a timestamped experiments/arithmetic-planner run",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out_dir = Path("experiments/arithmetic-planner") / stamp

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    (args.out_dir / "workflows").mkdir(exist_ok=True)

    selected_models = [config for config in MODELS if args.models is None or config.label in args.models]

    rows: list[dict] = []
    for config in selected_models:
        for case in EXPRESSIONS:
            for run_index in range(1, args.runs + 1):
                print(f"Running {config.label}, {case['case']}, run {run_index}...")
                rows.append(
                    _run_one(
                        config,
                        case,
                        run_index,
                        args.out_dir,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )
                )

    results_path = args.out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    _print_summary(rows)
    print(f"\nWrote {results_path}")
    print(f"Workflow files are in {args.out_dir / 'workflows'}")


if __name__ == "__main__":
    main()
