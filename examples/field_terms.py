# pyright: reportInvalidTypeForm=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportUnusedExpression=false, reportUnboundVariable=false, reportReturnType=false

"""Field-term guard example: cross-lifeline version consistency check.

A Publisher distributes a document to a Reviewer and a version tag to a
Gatekeeper.  After review, the Reviewer forwards the verdict and content to
the Gatekeeper — but NOT the version.

The Gatekeeper's guard combines:

  verdict_ok      — Reviewer approved the content
  version_matches — Reviewer held the same version as the Gatekeeper

Point (2) uses a field-term comparison:

    At[Reviewer].rev_version == Here.version

The Reviewer's version arrives at the Gatekeeper implicitly: the monitor
piggybacks each lifeline's field_view onto every message, so the verdict
message carries the Reviewer's current variable snapshot for free.

This cross-lifeline comparison of two runtime values cannot be expressed with
At[Reviewer](...) alone — that fixes one side at formula-construction time.

Three runs show the three outcomes:
  1. Matching versions, good content  → published
  2. Matching versions, short content → rejected by verdict
  3. Mismatched versions              → rejected by field-term guard

Run with:
    python examples/field_terms.py
"""

from zippergen import At, Here, Lifeline, Var, workflow
from zippergen.actions import pure

Publisher  = Lifeline("Publisher")
Reviewer   = Lifeline("Reviewer")
Gatekeeper = Lifeline("Gatekeeper")

rev_version = Var("rev_version", str)
version     = Var("version",     str)
content     = Var("content",     str)
verdict     = Var("verdict",     str)
result      = Var("result",      str)


@pure
def review(content: str) -> str:
    return "approved" if len(content) > 10 else "rejected"


@pure
def publish(content: str) -> str:
    return f"Published: {content}"


@pure
def reject_verdict(verdict: str) -> str:
    return f"Rejected: reviewer verdict was '{verdict}'"


@pure
def reject_version(version: str) -> str:
    return f"Rejected: version mismatch (Gatekeeper holds {version})"


# Guard 1 — local field term: the received verdict is approved.
verdict_ok = Here.verdict == "approved"

# Guard 2 — field term: the Reviewer's version matches the Gatekeeper's.
# At[Reviewer].rev_version is the value of rev_version at the Reviewer's
# latest causally visible event.  Here.version is the Gatekeeper's current
# local version at the guard event.  The version is never explicitly sent
# from Reviewer to Gatekeeper.
version_matches = At[Reviewer].rev_version == Here.version

publish_guard = verdict_ok & version_matches


@workflow
def review_gate(rev: str @ Publisher, gk: str @ Publisher, doc: str @ Publisher) -> str:
    # Publisher sends the document to Reviewer and the version to Gatekeeper.
    # rev and gk may differ — e.g. Gatekeeper was updated after Reviewer started.
    Publisher(rev, doc) >> Reviewer(rev_version, content)
    Publisher(gk)       >> Gatekeeper(version)

    # Reviewer evaluates and forwards verdict + content — version is NOT included.
    Reviewer: (verdict,) = review(content)
    Reviewer(verdict, content) >> Gatekeeper(verdict, content)

    # Gatekeeper decides: both guards must hold.
    if publish_guard @ Gatekeeper:
        Gatekeeper: (result,) = publish(content)
    else:
        if verdict_ok @ Gatekeeper:
            # Verdict was fine but versions differ — caught by field term.
            Gatekeeper: (result,) = reject_version(version)
        else:
            Gatekeeper: (result,) = reject_verdict(verdict)

    return result @ Gatekeeper


if __name__ == "__main__":
    review_gate.configure(llms="mock", ui=True)

    print("Run 1 — matching versions, sufficient content:")
    r = review_gate(rev="v1.0", gk="v1.0", doc="A well-written document ready for release")
    print(f"  → {r}\n")

    print("Run 2 — matching versions, content too short:")
    r = review_gate(rev="v1.0", gk="v1.0", doc="Draft")
    print(f"  → {r}\n")

    print("Run 3 — version mismatch (Gatekeeper updated to v2.0, Reviewer still on v1.0):")
    r = review_gate(rev="v1.0", gk="v2.0", doc="A well-written document ready for release")
    print(f"  → {r}\n")

    input("ZipperChat is running at http://localhost:8765\nPress Enter to stop.\n")
