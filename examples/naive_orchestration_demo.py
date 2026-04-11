#!/usr/bin/env python3
"""
Naive orchestration demo — no frameworks, pure Python.

Shows how a simple coordination bug causes a deadlock between agents.

Modes:
  good   — correct protocol: Orchestrator sends review_request, waits
            for review_result, then sends execute_plan
  bad    — bug: Orchestrator polls for review_result before sending
            review_request → circular wait → all agents stuck
  random — random choice between good and bad each run

Usage:
  python naive_orchestration_demo.py [good|bad|random]
"""

import threading
import queue
import time
import random
import sys

TIMEOUT = 2.0  # seconds before an agent declares itself stuck


# ── Agents ────────────────────────────────────────────────────────────────────

def planner(q_orch):
    print("[Planner]       →  Orchestrator : plan")
    q_orch.put("plan")


def reviewer(q_in, q_out):
    print("[Reviewer]         waiting for review_request ...")
    try:
        msg = q_in.get(timeout=TIMEOUT)
        print(f"[Reviewer]      ←  Orchestrator : {msg}")
        time.sleep(0.05)
        print("[Reviewer]      →  Orchestrator : review_result")
        q_out.put("review_result")
    except queue.Empty:
        print("[Reviewer]         STUCK — review_request never arrived")


def executor(q_in):
    print("[Executor]         waiting for execute_plan ...")
    try:
        msg = q_in.get(timeout=TIMEOUT)
        print(f"[Executor]      ←  Orchestrator : {msg}")
        print("[Executor]         done.")
    except queue.Empty:
        print("[Executor]         STUCK — execute_plan never arrived")


def orchestrator(q_in, q_rev_out, q_rev_in, q_exec, buggy):
    msg = q_in.get()
    print(f"[Orchestrator]  ←  Planner      : {msg}")
    time.sleep(0.05)

    if buggy:
        # BUG: polls for review_result *before* sending review_request.
        # Orchestrator and Reviewer now wait on each other forever.
        print("[Orchestrator]     waiting for review_result ...")
        try:
            result = q_rev_in.get(timeout=TIMEOUT)
            print(f"[Orchestrator]  ←  Reviewer     : {result}")
            q_exec.put("execute_plan")
        except queue.Empty:
            print("[Orchestrator]     STUCK — review_result never arrived")
            print()
            print("  >>> SYSTEM STUCK: inconsistent branch execution <<<")
    else:
        print("[Orchestrator]  →  Reviewer     : review_request")
        q_rev_out.put("review_request")
        result = q_rev_in.get(timeout=TIMEOUT)
        print(f"[Orchestrator]  ←  Reviewer     : {result}")
        print("[Orchestrator]  →  Executor     : execute_plan")
        q_exec.put("execute_plan")
        print("[Orchestrator]     done.")


# ── Runner ────────────────────────────────────────────────────────────────────

def run(mode: str):
    if mode == "random":
        buggy = random.random() < 0.5
        label = "random → " + ("BUG" if buggy else "OK")
    else:
        buggy = mode == "bad"
        label = mode.upper()

    print(f"\n{'─' * 52}")
    print(f"  {label}")
    print(f"{'─' * 52}")

    q_orch    = queue.Queue()
    q_rev_out = queue.Queue()   # Orchestrator → Reviewer
    q_rev_in  = queue.Queue()   # Reviewer     → Orchestrator
    q_exec    = queue.Queue()

    threads = [
        threading.Thread(target=planner,      args=(q_orch,),
                         daemon=True, name="Planner"),
        threading.Thread(target=reviewer,      args=(q_rev_out, q_rev_in),
                         daemon=True, name="Reviewer"),
        threading.Thread(target=executor,      args=(q_exec,),
                         daemon=True, name="Executor"),
        threading.Thread(target=orchestrator,
                         args=(q_orch, q_rev_out, q_rev_in, q_exec, buggy),
                         daemon=True, name="Orchestrator"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=TIMEOUT + 0.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "random"
    if mode not in ("good", "bad", "random"):
        print(f"Usage: {sys.argv[0]} [good|bad|random]")
        sys.exit(1)

    if mode == "random":
        for i in range(1, 6):
            print(f"\nRun {i}/5", end="")
            run("random")
            time.sleep(0.1)
    else:
        run(mode)


if __name__ == "__main__":
    main()
