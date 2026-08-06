# Studio lifecycle reference

Short tables describing how a ZipperGen project behaves. Each table answers one
question. For explanation and examples, see the manual.

## Project files

| file | written by | in git | holds |
|---|---|---|---|
| `specification.md` | you, or `workflow refine-spec` | yes | what the workflow should do |
| `workflow.py` (any name) | `workflow implement`, or you | yes | the implementation |
| `zippergen.toml` | `project init`, `model setup`, `connector setup` | yes | entry point, specification file, shared settings |
| `zippergen.lock` | `workflow implement` only | yes | which specification version the code came from |
| `.zippergen/` | Studio | no | the refinement buffer and other scratch |
| `ZIPPERGEN_HOME` | Studio | no | keys, endpoints, runs, deployments, trash |

Everything in the first four rows travels with the project. A clone needs only
the keys and endpoints for the machine it runs on.

## Workflow state

Two rows per command: the first is what must be true before, the second what is
true after.

`b` means any value. `↓` means unchanged.

`implementation` is one of `absent`, `stale`, `current`, `external`. It is
worked out from the files, never stored as a flag.

| | specification | implementation | refinement |
|---|---|---|---|
| **workflow edit-spec** | b | b | b |
| → | present | `stale` ¹ | ↓ |
| | | | |
| **workflow edit-refinement** | present | b | b |
| → | present | ↓ | present ² |
| | | | |
| **workflow refine-spec** | present | b | present |
| → | present | `stale` ¹ | absent |
| | | | |
| **workflow implement** | present | b | absent |
| → | present | `current` | ↓ |
| | | | |
| **workflow import** | b | b ³ | b |
| → | ↓ | `external` | ↓ |
| | | | |
| **deploy** | present | not `absent`, not `stale` ⁴ | b |
| → | ↓ | ↓ | ↓ |

¹ `absent` stays `absent`, and `external` stays `external`. Neither was
generated from this specification, so neither can go stale. Anything else
becomes `stale` — and stays `current` if the edit changed nothing, because the
value is worked out by comparing fingerprints rather than being set.

² Or absent if you empty the buffer, which discards the refinement. This
matches `git commit` stopping when you leave the message empty.

³ One project holds one workflow. Importing over an existing implementation
replaces it and asks first.

⁴ `current` deploys. `external` deploys after a warning that the code was not
generated from the specification. `stale` and `absent` are blocked. The rule is:
block when something is known to be wrong, warn when it is only unknown.

**One thing only.** Only `workflow implement` produces `current`. Editing the
specification can only move a generated implementation away from it.

## What to do next

| when | next |
|---|---|
| no specification | `workflow edit-spec` |
| a refinement is waiting | `workflow refine-spec` |
| implementation is `external` | `workflow edit-spec` · `workflow implement` |
| implementation is `absent` or `stale` | `workflow implement` |
| otherwise | `run` · `deploy` |

## Git

Studio does not manage branches, remotes or merges. It tracks one fact: **would
a clone of this repository behave the same as your working copy?**

The files that decide this are `specification.md`, the implementation files
listed in `zippergen.lock`, the lock itself, and `zippergen.toml` when
implementation changed it. Together they are the *commit unit*.

| | clone matches your copy |
|---|---|
| commit offer after `workflow implement` | becomes yes |
| any change to a file in the commit unit | becomes no |
| `workflow status`, `deploy` | warn when no, never block |

If the project is not a git repository, none of this appears.

## What each machine must supply

Settings that describe one machine are never committed. A clone reports exactly
what is missing.

| needed by | requirement | how it is checked |
|---|---|---|
| `run` | a model for every participant and action that uses one | resolved, then the endpoint is reached |
| `run` | a connector configuration for every connector requirement | resolved, then the service is reached |
| `run` | the keys those need | read where they are used |
| `deploy` | the same models and connectors | resolved only |
| `deploy` | each connector has passed `connector config check` | the stored result is read |
| `deploy` | an implementation that is not `absent` or `stale` | worked out from the files |
| `deploy` | a coding assistant, if the workflow has `@assistant` actions | looked up on `PATH` |

`deploy` does not reach any endpoint. A deployment is often prepared before it
is started, and a local model server may come up beside it, so an endpoint that
is unreachable right now is not a reason to refuse. What must hold is that the
machine is configured: every participant has a model, every requirement has a
connector, and the assistant is installed.

`project` lists what is still missing on this machine.

## Where each command belongs

Studio has 87 commands in six areas. The tables above describe the **workflow**
area only; the others have their own behaviour and are documented in the manual.

| area | count | what it covers |
|---|---|---|
| `deploy` | 16 | bundles, services, durable state, storage, logs |
| `workflow` | 14 | specification, implementation, views, validation |
| `model` | 9 | providers, named configurations, assignments |
| `connector` | 9 | provider credentials, resources, human-action routes |
| `run` · `resume` · `runs` | 7 | durable development execution |
| `project` · `settings` · `editor` · `language` · `studio` · `ask` · `plan` · `current` · `edit` · `help` | 32 | the project itself and Studio's own behaviour |

Within the workflow area, each command either changes state or answers a
question. A workflow command that does neither should probably not exist.

| kind | commands |
|---|---|
| changes state | `edit-spec`, `edit-refinement`, `refine-spec`, `implement`, `import` |
| answers a question | `validate`, `show`, `diff`, `status`, `list`, `files`, `history`, `path` |

Commands that answer a question change nothing, so they have no "after" row.
They do have requirements:

| command | needs |
|---|---|
| `workflow validate` | an implementation. It checks that the code loads and projects, not that it matches the specification. |
| `workflow show`, `workflow diff` | an implementation |
| `workflow status`, `workflow list` | nothing |
