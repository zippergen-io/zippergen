"""Static lowering and role support for dynamically bound spawn."""

import pytest

from zippergen import Json, Lifeline, Var, all_except, branch, fragment, parallel, pure, spawn, workflow
from zippergen.availability import workflow_availability_error
from zippergen.control import program_fingerprint
from zippergen.projection import project, project_templates
from zippergen.syntax import (
    BoundForkStmt, BoundReceiveStmt, BoundSelectStmt, BoundSendStmt,
    SeqStmt, SpawnStmt, _all_workflow_lifelines, participation_set,
)

Manager = Lifeline("Manager")
Reviewer = Lifeline("Reviewer")
IT = Lifeline("IT")
bindings = Var("bindings", Json)
request = Var("request", str)
approval = Var("approval", bool)
stale = Var("stale", str, default="old")


@pure
def select_binding() -> Json:
    return {"Manager": "m", "Reviewer": "r", "IT": "i"}


@pure
def approve(request: str) -> bool:
    return True


@pure
def use_stale(stale: str) -> bool:
    return True


@fragment
def review():
    Manager(request) >> Reviewer(request)
    Reviewer: approval = approve(request)
    Reviewer(approval) >> IT(approval)


@workflow
def bound():
    Manager: bindings = select_binding()
    spawn(review, bindings=bindings @ Manager,
          retain=(Manager, IT), fresh="all")


def _leaves(node):
    if isinstance(node, SeqStmt):
        return [*_leaves(node.first), *_leaves(node.second)]
    return [node]


def test_bound_spawn_lowers_owner_and_retained_roles():
    statement = bound.body.second
    assert isinstance(statement, SpawnStmt)
    assert {role.name for role in participation_set(statement)} == {"Manager", "IT"}
    assert {field.lifeline.name for field in statement.fresh} == {"Manager", "IT"}
    assert [role.name for role in _all_workflow_lifelines(bound)] == [
        "Manager", "IT", "Reviewer"
    ]
    manager = _leaves(project(bound, Manager))
    assert [type(node) for node in manager[1:]] == [
        BoundSelectStmt, BoundSendStmt, BoundSendStmt, BoundForkStmt
    ]
    assert [node.recipient.name for node in manager[2:4]] == ["IT", "Reviewer"]
    it = _leaves(project(bound, IT))
    assert [type(node) for node in it] == [BoundReceiveStmt, BoundForkStmt]
    assert project(bound, Reviewer).__class__.__name__ == "EmptyStmt"
    templates = project_templates(bound)
    reviewer = templates[("0.1", "Reviewer")]
    assert reviewer.recruited and reviewer.fresh == ()
    assert reviewer.template.startswith("bound-v1:0.1:")
    assert program_fingerprint({"child": manager[-1].body}) == program_fingerprint(
        {"child": templates[("0.1", "Manager")].program}
    )


def test_recruited_role_cannot_read_preexisting_default():
    @workflow
    def invalid():
        Manager: bindings = select_binding()
        with spawn(bindings=bindings @ Manager, retain=(), fresh="all"):
            Reviewer: approval = use_stale(stale)

    assert "not definitely have" in workflow_availability_error(invalid)


def test_recruited_exclusion_is_rejected():
    with pytest.raises(ValueError, match="recruited"):
        @workflow
        def invalid():
            Manager: bindings = select_binding()
            spawn(review, bindings=bindings @ Manager, retain=(Manager, IT),
                  fresh=all_except(Reviewer.request))


def test_unknown_retention_and_explicit_recruit_fresh_fail():
    with pytest.raises(ValueError, match="distinct child roles"):
        @workflow
        def unknown_retention():
            Manager: bindings = select_binding()
            spawn(review, bindings=bindings @ Manager, retain=(Manager, Lifeline("Other")))

    with pytest.raises(ValueError, match="retained roles only"):
        @workflow
        def fresh_recruit():
            Manager: bindings = select_binding()
            spawn(review, bindings=bindings @ Manager, retain=(Manager, IT),
                  fresh=(Reviewer.request,))


def test_nested_recruits_do_not_enter_outer_support_or_freshness():
    @workflow
    def nested():
        with spawn(fresh="all"):
            Manager: bindings = select_binding()
            spawn(review, bindings=bindings @ Manager,
                  retain=(Manager,), fresh="all")

    outer = nested.body
    assert {role.name for role in participation_set(outer.body)} == {"Manager"}
    assert {field.lifeline.name for field in outer.fresh} == {"Manager"}
    assert "Reviewer" not in {field.lifeline.name for field in outer.fresh}
    assert ("0.0.1", "Reviewer") in project_templates(nested)


def test_parallel_bound_sites_are_distinct():
    @workflow
    def parallel_reviews():
        Manager: bindings = select_binding()
        with parallel:
            with branch:
                spawn(review, bindings=bindings @ Manager,
                      retain=(Manager, IT), fresh="all")
            with branch:
                spawn(review, bindings=bindings @ Manager,
                      retain=(Manager, IT), fresh="all")

    templates = project_templates(parallel_reviews)
    sites = {site for site, role in templates if role == "Reviewer"}
    assert len(sites) == 2
    assert len({templates[(site, "Reviewer")].template for site in sites}) == 2


@fragment
def reviewer_only():
    Reviewer: approval = approve("request")


@workflow
def owner_absent_from_child():
    Manager: bindings = select_binding()
    spawn(reviewer_only, bindings=bindings @ Manager, retain=())


@fragment
def manager_only():
    Manager: approval = approve("request")


@workflow
def owner_recruited():
    Manager: bindings = select_binding()
    spawn(manager_only, bindings=bindings @ Manager, retain=())


def test_owner_absent_and_owner_recruited_get_correct_parent_code():
    absent = _leaves(project(owner_absent_from_child, Manager))
    assert [type(item) for item in absent[1:]] == [BoundSelectStmt, BoundSendStmt]
    assert absent[-1].recipient == Reviewer and absent[-1].kind == "invite"
    recruited = _leaves(project(owner_recruited, Manager))
    assert [type(item) for item in recruited[1:]] == [BoundSelectStmt, BoundSendStmt]
    assert recruited[-1].recipient == Manager and recruited[-1].kind == "invite"
    assert project_templates(owner_recruited)[("0.1", "Manager")].recruited


def test_binding_read_must_be_owner_local():
    @workflow
    def nonlocal_binding():
        IT: bindings = select_binding()
        spawn(review, bindings=bindings @ Manager, retain=(Manager, IT))

    error = workflow_availability_error(nonlocal_binding)
    assert error is not None and "spawn binding on 'Manager'" in error
