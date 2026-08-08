# ZipperGen repository guidance

For requests to create, extend, inspect, validate, compare, or prepare the
deployment of a ZipperGen workflow, read
`.agents/skills/zippergen-workflows/SKILL.md` completely and follow it. Read the linked
`references/dsl-and-cli.md` before editing workflow code.

That directory mirrors `src/zippergen/skills/zippergen-workflows/`, which is
what ships in the package. Edit the packaged copy and mirror it; a test asserts
the two are identical. Outside a checkout the same content comes from
`zippergen skill`.

Keep workflow transformations code-first. Validate generated protocols, inspect
the relevant global and local code views, and use a semantic snapshot/diff for
changes to existing workflows.
