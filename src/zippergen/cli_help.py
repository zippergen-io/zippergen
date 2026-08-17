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


def _advertised(subparsers) -> list:
    """Return the subcommands argparse itself offers, in registration order.

    A parser added without ``help=`` gets no pseudo-action, which is how
    argparse hides a subcommand that still parses. The tree follows the same
    signal, so a hidden verb cannot reappear here.
    """

    listed = {
        action.dest for action in getattr(subparsers, "_choices_actions", [])
    }
    if not listed:
        return list(subparsers.choices.items())
    return [
        (name, parser)
        for name, parser in subparsers.choices.items()
        if name in listed
    ]


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
        children = _advertised(nested)
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
