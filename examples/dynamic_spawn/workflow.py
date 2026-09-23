"""Two independent access reviews with reviewers selected at runtime.

Run this file directly for a temporary SQLite-backed demonstration. All actions
are local fakes; there are no credentials, models or external side effects.
"""
from zippergen import Json, Lifeline, Var, fragment, pure, run_sqlite, spawn, workflow

Manager, Reviewer, IT = (Lifeline(name) for name in ("Manager", "Reviewer", "IT"))
index = Var("index", int, default=0)
request = Var("request", str)
approved = Var("approved", bool)
receipt = Var("receipt", str)


@pure
def next_request(index: int) -> str:
    return ("read:alpha", "write:beta")[index]


@pure
def select_reviewer(index: int) -> Json:
    return {
        "Manager": "Manager",
        "Reviewer": ("reviewer-east", "reviewer-west")[index],
        "IT": "IT",
    }


@pure
def assess(request: str) -> bool:
    return request.startswith("read:")


@pure
def provision(request: str) -> str:
    return "provisioned:" + request


@pure
def advance(index: int) -> int:
    return index + 1


@fragment
def AccessReview():
    Manager(request) >> Reviewer(request)
    Reviewer: approved = assess(request)
    if approved @ Reviewer:
        Reviewer(request) >> IT(request)
        IT: receipt = provision(request)


@workflow
def access_reviews() -> int:
    while (index < 2) @ Manager:
        Manager: request = next_request(index)
        Manager: bindings = select_reviewer(index)
        spawn(
            AccessReview,
            bindings=bindings @ Manager,
            retain=(Manager, IT),
            fresh="all",
        )
        Manager: index = advance(index)
    return index @ Manager


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="zippergen-access-review-") as directory:
        result = run_sqlite(
            access_reviews,
            store_path=str(Path(directory) / "run.sqlite"),
            endpoints={"reviewer-east", "reviewer-west"},
            verbose=True,
        )
        print(f"Dispatched {result} reviews; all child sessions finished.")
