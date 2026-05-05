"""
CLI human backend for HumanAction.

make_cli_human_backend() returns a callable with the same signature as
llm_backend: (action: HumanAction, inputs: dict) -> dict.
"""

from __future__ import annotations

__all__ = ["make_cli_human_backend"]


def make_cli_human_backend():
    """
    Return a human backend that blocks on stdin.

    - bool output: prompts [y/n], loops until valid, returns True/False.
    - str output without options: reads one line.
    - str output with options: prints numbered list, loops until valid selection.
    """
    def backend(action, inputs: dict) -> dict:
        from zippergen.syntax import HumanAction
        assert isinstance(action, HumanAction)

        prompt = action.prompt.format(**inputs)

        if action.output_type is bool:
            while True:
                raw = input(f"{prompt} [y/n]: ").strip().lower()
                if raw in ("y", "yes"):
                    value: object = True
                    break
                if raw in ("n", "no"):
                    value = False
                    break
                print("Please enter 'y' or 'n'.")

        elif action.options is not None:
            print(prompt)
            for i, opt in enumerate(action.options, 1):
                print(f"  {i}. {opt}")
            while True:
                raw = input("Enter number: ").strip()
                if raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(action.options):
                        value = action.options[idx]
                        break
                print(f"Please enter a number between 1 and {len(action.options)}.")

        else:
            value = input(f"{prompt}: ").strip()

        return {action.output: value}

    return backend
