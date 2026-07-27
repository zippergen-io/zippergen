# Retire ZipperChat from the shipped product

Date: 27 July 2026

ZipperChat was a browser viewer for execution traces and Message Sequence
Charts. It also contained an early browser path for completing human tasks
stored in SQLite.

We removed it from the package for three reasons:

1. Studio now provides the main development and operating environment.
2. Local projection inspection stays bounded by program size. A growing
   Message Sequence Chart does not.
3. One product name and one code-first interface are easier to understand.

The useful ideas are not lost:

- Human tasks remain durable in SQLite.
- Studio and the CLI can inspect and complete those tasks.
- Telegram can deliver approvals without requiring the approver to install
  ZipperGen.
- Git history preserves the former browser approval implementation if a future
  web approval client is needed.
- Message Sequence Chart rendering can live with research papers, talks, and
  website material. It is not part of the Python package.

This decision removes `zipperchat`, the `ui` and `show_decisions` workflow
options, their CLI flags, browser replay state, and the static browser demo.
At the current alpha stage, these interfaces are removed rather than retained
as ignored compatibility options.
