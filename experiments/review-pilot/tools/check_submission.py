#!/usr/bin/env python3
"""Rebuild one proof against a pristine frozen packet and audit its route."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import packet
import route_policy

ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
FORBIDDEN_SOURCE = (
    r"\baxiom\b", r"\bunsafe\b", r"\bnative_decide\b",
    r"\badmit\b", r"\bsorry\b", r"\bopaque\b", r"#eval\b",
    r"\brun_cmd\b", r"\brun_tac\b", r"\belab\b", r"\bmacro\b",
    r"\bsyntax\b", r"\bIO\b", r"\bimplemented_by\b",
)


def strip_comments(source: str) -> str:
    out = []
    i = 0
    depth = 0
    quoted = False
    while i < len(source):
        pair = source[i:i + 2]
        char = source[i]
        if depth:
            if pair == "/-":
                depth += 1
                i += 2
            elif pair == "-/":
                depth -= 1
                i += 2
            else:
                if char == "\n":
                    out.append("\n")
                i += 1
        elif quoted:
            out.append(char)
            if char == "\\" and i + 1 < len(source):
                i += 1
                out.append(source[i])
            elif char == '"':
                quoted = False
            i += 1
        elif pair == "/-":
            depth = 1
            i += 2
        elif pair == "--":
            while i < len(source) and source[i] != "\n":
                i += 1
        else:
            out.append(char)
            quoted = char == '"'
            i += 1
    return "".join(out)


def source_ok(arm: str, path: Path) -> tuple[bool, list[str]]:
    code = strip_comments(path.read_text())
    problems = []
    statement = re.search(r"\btheorem\s+submission\s*:\s*(.*?)\s*:=", code, re.S)
    if statement is None or " ".join(statement.group(1).split()) != packet.GOAL:
        problems.append("wrong theorem statement")
    problems.extend(pattern for pattern in FORBIDDEN_SOURCE if re.search(pattern, code))
    allowed = {Path(name).stem for name in packet.given_files(arm)}
    for module in re.findall(r"^\s*import\s+(\S+)", code, re.M):
        if module in allowed or module == "Lean" or re.fullmatch(r"(?:Init|Std)(?:\..+)?", module):
            continue
        if module.startswith(("MSCAgents.", "ZipperGenSpawn.", "ZipperGenParallel.")):
            continue
        problems.append(f"bad import: {module}")
    return not problems, problems


def check(k: int, arm: str, solution: Path) -> dict:
    packet.verify_frozen()
    if solution.is_symlink():
        return {"proved": False, "reason": "symlink submission"}
    okay, problems = source_ok(arm, solution)
    result = {"k": k, "arm": arm, "statement": packet.GOAL,
              "source_ok": okay, "source_problems": problems,
              "route_rule": "direct-only" if arm == "local" else "unrestricted"}
    if not okay:
        result["proved"] = False
        return result

    with tempfile.TemporaryDirectory(prefix="zg-review-check-") as raw:
        workspace = Path(raw) / "work"
        packet.build(k, arm, workspace)
        shutil.copy2(solution, workspace / "Solution.lean")
        check_text = (
            "import Solution\nimport Lean\n"
            f"theorem checked_submission : _root_.{packet.GOAL} := _root_.submission\n"
            "#print axioms checked_submission\n"
            + route_policy.submission_audit(arm == "local")
        )
        (workspace / "Check.lean").write_text(check_text)
        started = time.monotonic()
        proc = subprocess.run(["lake", "build", "Given", "Solution", "Check"],
                              cwd=workspace, capture_output=True, text=True,
                              timeout=3600)
        log = proc.stdout + proc.stderr
        found = re.search(r"'checked_submission' depends on axioms: \[([^\]]*)\]", log)
        axioms = ({part.strip() for part in found.group(1).split(",")}
                  if found else None)
        result.update({
            "build_ok": proc.returncode == 0,
            "build_seconds": round(time.monotonic() - started, 1),
            "axioms": sorted(axioms) if axioms is not None else None,
            "banned_dependencies": re.findall(r"Forbidden global proof dependency: ([^\s]+)", log),
            "errors": [line for line in log.splitlines() if "error:" in line][:15],
            "proved": proc.returncode == 0 and axioms is not None and axioms <= ALLOWED_AXIOMS,
        })
    return result


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit("usage: check_submission.py K global|local Solution.lean [out.json]")
    result = check(int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
    if len(sys.argv) == 5:
        Path(sys.argv[4]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["proved"] else 1)
