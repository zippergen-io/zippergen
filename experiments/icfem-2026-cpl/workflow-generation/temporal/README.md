# Temporal workflow-generation task

This task keeps the four-stage chain of the remote-field task and adds three
successive status updates at L0. The final status is checking in one accepting
case, so a comparison with the current status is insufficient. The case
passed, failed, passed distinguishes local Since from a condition that requires
a pass and the complete absence of failure.

The task uses the same Codex CLI version, model, reasoning effort, options, and
prompt as the remote-field task. Each run starts in a fresh project created by
zippergen init. The agent receives specification.md and the standard ZipperGen
skill. It does not receive reference.json.
