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
| `run` | a model for every participant and action that uses one | resolved, then probed |
| `run` | a connector for every connector requirement | resolved, then probed |
| `run` | the keys those need | read where they are used |
| `deploy --no-start` | the same models and connectors | resolved only |
| `deploy --no-start` | the key each connector needs | read from this machine |
| `deploy --no-start` | an implementation that is not `absent` or `stale` | worked out from the files |
| `deploy --no-start` | a coding assistant, if the workflow has `@assistant` actions | looked up on `PATH` |
| `deploy` · `deploy start` | everything above | models and connectors are probed |

## Checks

A check is always live, and a live failure always stops you.

`model config check` and `connector config check` reach the real endpoint and
print what they find. They save nothing. There is no stored health anywhere, so
nothing can go stale, and nothing has to be kept in step between machines.

This is why listings and checks are separate:

| command | question | cost |
|---|---|---|
| `model assignments` · `connector assignments` | what is assigned? | instant, works offline |
| `model config check` · `connector config check` | does it work? | needs the network |

Preparing a deployment and starting one are different moments. `deploy
--no-start` only writes files, and it may be started days later, so an endpoint
that is unreachable now says nothing and the routes are merely resolved.
`deploy` and `deploy start` are about to make real calls, so every model and
connector is probed and a failure stops them. `deploy start` reads the routes
the deployment itself recorded, not whichever workflow happens to be selected.

Blocking is the same everywhere — on your laptop, on a server, in a script.
Nothing asks a question that a script cannot answer.

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
