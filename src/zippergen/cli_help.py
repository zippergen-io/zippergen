"""Parser-derived command overview for humans and coding agents."""

from __future__ import annotations

import argparse
from collections.abc import Collection


def _subparsers(parser: argparse.ArgumentParser):
    return next(
        (
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ),
        None,
    )


def render_command_tree(
    parser: argparse.ArgumentParser,
    *,
    hidden: Collection[str] = (),
) -> str:
    """Render every registered public command from the real parser tree."""

    root = _subparsers(parser)
    if root is None:
        return ""
    entries = [
        (name, child)
        for name, child in root.choices.items()
        if name not in hidden
    ]
    lines = ["Command tree:"]
    for index, (name, child) in enumerate(entries):
        last = index == len(entries) - 1
        branch = "└──" if last else "├──"
        lines.append(f"{branch} {name}")
        nested = _subparsers(child)
        if nested is None:
            continue
        prefix = "    " if last else "│   "
        children = list(nested.choices.items())
        for child_index, (action, action_parser) in enumerate(children):
            action_last = child_index == len(children) - 1
            action_branch = "└──" if action_last else "├──"
            lines.append(f"{prefix}{action_branch} {action}")
            grandchildren = _subparsers(action_parser)
            if grandchildren is None:
                continue
            nested_prefix = prefix + ("    " if action_last else "│   ")
            grandchild_items = list(grandchildren.choices.items())
            for grandchild_index, (leaf, leaf_parser) in enumerate(
                grandchild_items
            ):
                leaf_branch = (
                    "└──"
                    if grandchild_index == len(grandchild_items) - 1
                    else "├──"
                )
                lines.append(f"{nested_prefix}{leaf_branch} {leaf}")
    return "\n".join(lines)


__all__ = ["render_command_tree"]
