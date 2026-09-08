"""Recheck the three generated temporal workflows."""

from __future__ import annotations

import argparse
import json
from itertools import product
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
EXPECTED_MESSAGES = [
    "L0(token) >> L1(token)",
    "L1(token) >> L2(token)",
    "L2(token) >> L3(token)",
]
STATUSES = ("passed", "checking", "failed")


def zg(executable: str, project: Path, home: Path, *args: str) -> str:
    env = os.environ.copy()
    env["ZIPPERGEN_HOME"] = str(home)
    return subprocess.run(
        [executable, *args],
        cwd=project,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def assess(executable: str, run: int) -> dict[str, object]:
    source = ROOT / "results" / f"run{run}"
    with tempfile.TemporaryDirectory(prefix=f"cpl-temporal-run{run}-") as raw:
        temporary = Path(raw)
        project = temporary / "project"
        shutil.copytree(source, project)
        home = temporary / "zippergen-home"

        validation = json.loads(zg(executable, project, home, "validate", "--json"))
        communication = zg(executable, project, home, "show", "--communications")
        messages = [
            line.strip() for line in communication.splitlines() if ">>" in line
        ]
        projections = validation["projections"]
        l0_updates = projections["L0"].count("status = record_status(")
        guard = next(
            line.strip()
            for line in projections["L3"].splitlines()
            if line.strip().startswith("if ")
        )

        failures = []
        histories_checked = 0
        for statuses in product(STATUSES, repeat=3):
            decisive = [status for status in statuses if status != "checking"]
            expected = (
                "accept" if decisive and decisive[-1] == "passed" else "reject"
            )
            command = ["run"]
            for name, value in zip(
                ("first_status", "second_status", "third_status"), statuses
            ):
                command.extend(("--input", f"{name}={value}"))
            command.extend(("--input", "token=opaque"))
            actual = json.loads(zg(executable, project, home, *command))["result"]
            histories_checked += 1
            if actual != expected:
                failures.append({
                    "statuses": list(statuses),
                    "expected": expected,
                    "actual": actual,
                })

        return {
            "run": run,
            "valid": validation["valid"],
            "messages_match": messages == EXPECTED_MESSAGES,
            "three_status_updates": l0_updates == 3,
            "l3_guard": guard,
            "histories_checked": histories_checked,
            "failures": failures,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zippergen",
        default="zippergen",
        help="ZipperGen executable to use (default: zippergen on PATH)",
    )
    args = parser.parse_args()
    results = [assess(args.zippergen, run) for run in range(1, 4)]
    print(json.dumps({"runs": results}, indent=2))
    passed = all(
        result["valid"]
        and result["messages_match"]
        and result["three_status_updates"]
        and result["histories_checked"] == 27
        and not result["failures"]
        for result in results
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
