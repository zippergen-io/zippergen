"""A workflow with a first-class repository-aware assistant action."""

from zippergen import Lifeline, assistant, workflow


Maintainer = Lifeline("Maintainer")


@assistant(
    instructions_file="examples/prompts/update_release_notes.md",
    workspace=".",
)
def update_release_notes(change: str) -> str: ...


@workflow
def assistant_maintenance(change: str @ Maintainer) -> str:
    Maintainer: report = update_release_notes(change)
    return report @ Maintainer
