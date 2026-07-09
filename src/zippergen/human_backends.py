"""
CLI human backend for HumanAction.

make_cli_human_backend() returns a callable with the same signature as
llm_backend: (action: HumanAction, inputs: dict) -> dict.
"""

from __future__ import annotations

__all__ = ["make_cli_human_backend", "make_sqlite_human_backend"]


def make_cli_human_backend():
    """
    Return a human backend that blocks on stdin.

    - confirm: prompts [y/n], loops until valid, returns True/False.
    - ack: shows notification, press Enter to acknowledge, returns True.
    - edit: shows pre-filled text, reads one line (empty = keep prefill).
    - select: prints numbered list, loops until valid selection.
    - input: free-form text input.
    """
    def backend(action, inputs: dict) -> dict:
        from zippergen.syntax import HumanAction
        assert isinstance(action, HumanAction)

        context_text = action.context.format(**inputs) if action.context else None
        instruction_text = action.instruction.format(**inputs) if action.instruction else None
        prefill_text = action.prefill.format(**inputs) if action.prefill else None

        parts = []
        if context_text:
            parts.append(context_text)
        if instruction_text:
            parts.append(instruction_text)
        display = "\n\n".join(parts)

        if action.kind == "confirm":
            while True:
                raw = input(f"{display} [y/n]: ").strip().lower()
                if raw in ("y", "yes"):
                    value: object = True
                    break
                if raw in ("n", "no"):
                    value = False
                    break
                print("Please enter 'y' or 'n'.")

        elif action.kind == "ack":
            input(f"{display} [press Enter to acknowledge]")
            value = True

        elif action.kind == "select":
            options = [o.strip() for o in (prefill_text or "").split("\n") if o.strip()]
            if options:
                print(display)
                for i, opt in enumerate(options, 1):
                    print(f"  {i}. {opt}")
                while True:
                    raw = input("Enter number: ").strip()
                    if raw.isdigit():
                        idx = int(raw) - 1
                        if 0 <= idx < len(options):
                            value = options[idx]
                            break
                    print(f"Please enter a number between 1 and {len(options)}.")
            else:
                value = input(f"{display}: ").strip()

        elif action.kind == "edit":
            if prefill_text:
                print(f"{display}\n[Current: {prefill_text!r}]")
                raw = input("Edit (empty to keep): ").strip()
                value = raw if raw else prefill_text
            else:
                value = input(f"{display}: ").strip()

        else:  # input
            value = input(f"{display}: ").strip()

        return {action.output: value}

    return backend


def make_sqlite_human_backend():
    """Return a marker backend for SQLite-owned human tasks.

    RoleRunner creates the durable task row and then waits for some external
    actor (CLI, UI, Telegram, email, etc.) to complete it in SQLite.
    """

    def backend(action, inputs: dict) -> dict:
        raise RuntimeError("SQLite human tasks are completed through the SQLite store.")

    setattr(backend, "uses_sqlite_human_tasks", True)
    return backend
