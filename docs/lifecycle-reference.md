# Studio lifecycle reference

Short tables describing how a ZipperGen project behaves. Each table answers one
question. For explanation and examples, see the manual.

## Project files

| file | written by | in git | holds |
|---|---|---|---|
| `specification.md` | you, or `workflow refine-spec` | yes | what the workflow should do |
| `workflow.py` (any name) | `workflow implement`, or you | yes | the implementation |
| `zippergen.toml` | `project init`, `model setup`, `connector setup` | yes | entry point, specification file, shared settings |
| `zippergen.lock` | `workflow implement` · `workflow adopt` | yes | that this specification and this code correspond |
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
| **workflow adopt** | b | `external` | absent |
| → | present ⁵ | `current` | ↓ |
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

⁵ Written from the code by a coding assistant, then opened for you to review
before the pair is recorded. Refuses unless the implementation is `external`,
so it can never overwrite a specification you are deliberately working ahead
of.

**Two ways in, one meaning.** `current` means the specification and the code
correspond — not that one produced the other. `workflow implement` makes the
code follow the specification; `workflow adopt` makes the specification
describe the code. Editing the specification can only move the pair apart.

**Not a round trip.** Both directions are lossy: adopting and then
re-implementing gives you different code. Adopt exists so imported code can
enter the refinement loop, not so code and specification can be regenerated
from each other.

## What to do next

First matching row wins.

| when | next |
|---|---|
| a refinement is waiting | `workflow refine-spec` |
| implementation is `external` | `workflow adopt` |
| no specification | `workflow edit-spec` |
| implementation is `absent` or `stale` | `workflow implement` |
| otherwise | `run` · `deploy` |

`external` is checked before a missing specification on purpose. Importing a
workflow leaves exactly that pair, and writing a specification by hand only to
regenerate the code would throw away what you imported.

## Runs

A development run is durable. It keeps its own SQLite store, so an interrupted
run continues rather than starting over.

One run is *current* at a time — the most recent one you started. Every `run`
subcommand acts on that one. `runs` lists all of them.

A run is in one of five states:

| state | meaning |
|---|---|
| `running` | executing now |
| `waiting` | stopped at a human action, waiting for an answer |
| `interrupted` | you pressed Ctrl-C |
| `failed` | a participant raised an error |
| `done` | finished, with a result |

`running` and `waiting` alternate as the workflow reaches and passes each human
action. The last three are final for that run — `interrupted` and `failed` can
be continued with `resume`, `done` cannot.

| command | needs | does |
|---|---|---|
| `run` | a workflow that loads · a model for every LLM participant · a connector for every requirement | starts a new run and makes it current |
| `resume` | a current run that is not `done` | continues it from where it stopped |
| `runs` | nothing | lists every run, current one marked |
| `run tasks` | a current run | shows decisions waiting for a person |
| `run approve` | a current run with a pending decision | answers one |
| `run trace` | a current run | shows recent events |
| `run inspect` | a current run | shows where each participant stands |

Models and connectors are probed before a run starts, and a failure stops it
before any inputs are collected. See *What each machine must supply*.

Only `run` and `resume` change anything. The other five answer questions.

**Why `resume` and not restart.** The store records what already happened, so
resuming re-enters the workflow at the point it stopped. Work already done is
not repeated, and an LLM call already made is not paid for twice.

## Deployments

A deployment is an installed copy of one workflow, with its own managed Python
environment, its own service registration, and its own durable store. A project
can have several, each with a name.

Preparing and starting are separate. `deploy NAME --no-start` writes everything
and stops; `deploy NAME` also starts the service.

| state | meaning |
|---|---|
| prepared | files written, service not installed |
| started | service installed and running |
| stopped | service installed, not running |
| removed | gone from active use |

| command | needs | does |
|---|---|---|
| `deploy NAME` | an implementation that is not `absent` or `stale` · every model and connector reachable | prepares, then starts |
| `deploy NAME --no-start` | the same, but nothing is probed | prepares only |
| `deploy start` · `deploy restart` | a prepared deployment · its recorded routes reachable | starts it |
| `deploy stop` | a started deployment | stops the service, keeps everything |
| `deploy remove [--purge]` | a deployment | deletes it; see below |
| `deploy storage compact` | a **stopped** deployment | drops events already past the replay floor |
| `deploy logs reset` | a deployment | archives the visible log and starts a new one |
| `deploy show` · `doctor` · `logs` · `trace` · `tasks` · `inspect` · `storage` | a deployment | answer questions, change nothing |
| `deploy approve` | a pending decision | answers one |
| `deploy list` | nothing | lists every deployment |

Only `deploy`, `start`, `restart`, `stop`, `remove`, `storage compact`,
`logs reset`, and `approve` change anything. The rest answer questions.

`deploy storage compact` refuses while the service is running. Compaction reads
the replay floors, and a running deployment can move them underneath it.

### What removal keeps

`deploy remove` deletes the deployment but keeps what cannot be got back:

| artifact | kept | why |
|---|---|---|
| durable store, WAL sidecars | yes | the record of what actually ran |
| deployment log | yes | the same |
| profile | yes | says what produced them |
| secrets | **no** | must not be left behind |
| managed environment | no | rebuilt by deploying again |
| source bundles | no | git has the source |
| service files, run script | no | regenerated |

`deploy remove --purge` keeps nothing, including the store. Use it when the
history itself should be gone.

What survives lands in `$ZIPPERGEN_HOME/trash/deployments/`, owner-readable
only. Nothing prunes it, so it is worth looking at occasionally.

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

Studio has 88 commands in six areas. The tables above cover the **workflow**,
**run**, and **deploy** areas — every part of Studio with real state. The
`model`, `connector`, and settings commands are configuration, and are
documented in the manual.

| area | count | what it covers |
|---|---|---|
| `deploy` | 16 | bundles, services, durable state, storage, logs |
| `workflow` | 15 | specification, implementation, views, validation |
| `model` | 9 | providers, named configurations, assignments |
| `connector` | 9 | provider credentials, resources, human-action routes |
| `run` · `resume` · `runs` | 7 | durable development execution |
| `project` · `settings` · `editor` · `language` · `studio` · `ask` · `plan` · `current` · `edit` · `help` | 32 | the project itself and Studio's own behaviour |

Within the workflow area, each command either changes state or answers a
question. A workflow command that does neither should probably not exist.

| kind | commands |
|---|---|
| changes state | `edit-spec`, `edit-refinement`, `refine-spec`, `adopt`, `implement`, `import` |
| answers a question | `validate`, `show`, `diff`, `status`, `list`, `files`, `history`, `path` |

Commands that answer a question change nothing, so they have no "after" row.
They do have requirements:

| command | needs |
|---|---|
| `workflow validate` | an implementation. It checks that the code loads and projects, not that it matches the specification. |
| `workflow show`, `workflow diff` | an implementation |
| `workflow status`, `workflow list` | nothing |
