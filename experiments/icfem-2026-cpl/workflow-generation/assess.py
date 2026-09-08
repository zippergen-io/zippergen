"""Recheck the three generated coin-chain workflows."""

from __future__ import annotations

import argparse
import json
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
    with tempfile.TemporaryDirectory(prefix=f"cpl-coin-run{run}-") as raw:
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
        owner_guard = "if At[L0].outcome == 'heads':" in projections["L3"]
        other_guards = any("if " in projections[name] for name in ("L0", "L1", "L2"))

        outcomes = {}
        for outcome in ("heads", "tails"):
            output = zg(
                executable,
                project,
                home,
                "run",
                "--input",
                f"outcome={outcome}",
                "--input",
                "token=opaque",
            )
            outcomes[outcome] = json.loads(output)["result"]

        return {
            "run": run,
            "valid": validation["valid"],
            "messages_match": messages == EXPECTED_MESSAGES,
            "l3_owns_expected_guard": owner_guard and not other_guards,
            "heads_result": outcomes["heads"],
            "tails_result": outcomes["tails"],
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
        and result["l3_owns_expected_guard"]
        and result["heads_result"] == "heads"
        and result["tails_result"] == "tails"
        for result in results
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
