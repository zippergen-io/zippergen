"""Static lowering and support for owner-allocated fresh child endpoints."""

import pytest

from zippergen import Lifeline, Var, all_except, branch, fragment, parallel, pure, spawn, workflow, skip
from zippergen.availability import workflow_availability_error
from zippergen.control import program_fingerprint
from zippergen.projection import project, project_templates
from zippergen.syntax import (
    BoundForkStmt, InitSendStmt,
    SeqStmt, SpawnStmt, _all_workflow_lifelines, participation_set,
)

Manager = Lifeline("Manager")
Reviewer = Lifeline("Reviewer")
IT = Lifeline("IT")
request = Var("request", str, default="request")
approval = Var("approval", bool)
stale = Var("stale", str, default="old")


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
def allocated():
    spawn(review, at=Manager, retain=(Manager, IT), fresh="all")


def _leaves(node):
    if isinstance(node, SeqStmt):
        return [*_leaves(node.first), *_leaves(node.second)]
    return [node]


def test_allocated_spawn_lowers_owner_and_retained_roles():
    statement = allocated.body
    assert isinstance(statement, SpawnStmt)
    assert statement.owner == Manager
    assert {role.name for role in participation_set(statement)} == {"Manager", "IT"}
    assert {field.lifeline.name for field in statement.fresh} == {"Manager", "IT"}
    assert [role.name for role in _all_workflow_lifelines(allocated)] == [
        "Manager", "IT", "Reviewer"
    ]
    manager = _leaves(project(allocated, Manager))
    assert [type(node) for node in manager] == [
        InitSendStmt, BoundForkStmt
    ]
    assert [node.recipient.name for node in manager[:1]] == ["Reviewer"]
    it = _leaves(project(allocated, IT))
    assert [type(node) for node in it] == [BoundForkStmt]
    assert project(allocated, Reviewer).__class__.__name__ == "EmptyStmt"
    templates = project_templates(allocated)
    reviewer = templates[("0", "Reviewer")]
    assert reviewer.fresh_endpoint and reviewer.fresh == ()
    assert reviewer.template.startswith("fresh-v2:0:")
    assert program_fingerprint({"child": manager[-1].body}) == program_fingerprint(
        {"child": templates[("0", "Manager")].program}
    )


def test_inherited_default_and_explicit_all_roles_remain_initiator_free():
    @workflow
    def implicit():
        spawn(review)

    @workflow
    def explicit():
        spawn(review, retain=(Manager, Reviewer, IT))

    assert implicit.body.owner is None and explicit.body.owner is None
    assert program_fingerprint({role.name: project(implicit, role)
                                for role in (Manager, Reviewer, IT)}) == \
           program_fingerprint({role.name: project(explicit, role)
                                for role in (Manager, Reviewer, IT)})


def test_fresh_endpoint_cannot_read_preexisting_default():
    @workflow
    def invalid():
        with spawn(at=Manager, retain=(Manager,), fresh="all"):
            skip(Manager)
            Reviewer: approval = use_stale(stale)

    assert "not definitely have" in workflow_availability_error(invalid)


def test_fresh_endpoint_exclusion_is_rejected():
    with pytest.raises(ValueError, match="fresh endpoint"):
        @workflow
        def invalid():
            spawn(review, at=Manager, retain=(Manager, IT),
                  fresh=all_except(Reviewer.request))


def test_invalid_retention_and_explicit_fresh_are_rejected():
    with pytest.raises(ValueError, match="distinct child roles"):
        @workflow
        def unknown_retention():
            spawn(review, at=Manager, retain=(Manager, Lifeline("Other")))

    with pytest.raises(ValueError, match="retained roles only"):
        @workflow
        def fresh_new_endpoint():
            spawn(review, at=Manager, retain=(Manager, IT),
                  fresh=(Reviewer.request,))

    with pytest.raises(ValueError, match="retains every child role"):
        @workflow
        def incomplete_inherited_retention():
            spawn(review, retain=(Manager, IT))


def test_nested_fresh_endpoints_do_not_enter_outer_support_or_freshness():
    @workflow
    def nested():
        with spawn(fresh="all"):
            spawn(review, at=Manager, retain=(Manager,), fresh="all")

    outer = nested.body
    assert {role.name for role in participation_set(outer.body)} == {"Manager"}
    assert {field.lifeline.name for field in outer.fresh} == {"Manager"}
    assert ("0.0", "Reviewer") in project_templates(nested)


def test_parallel_allocation_sites_are_distinct():
    @workflow
    def parallel_reviews():
        with parallel:
            with branch:
                spawn(review, at=Manager, retain=(Manager, IT), fresh="all")
            with branch:
                spawn(review, at=Manager, retain=(Manager, IT), fresh="all")

    templates = project_templates(parallel_reviews)
    sites = {site for site, role in templates if role == "Reviewer"}
    assert len(sites) == 2
    assert len({templates[(site, "Reviewer")].template for site in sites}) == 2


def test_retained_child_template_matches_fork_after_parent_decision():
    @workflow
    def decision_then_spawn():
        if (request == "request") @ Manager:
            Manager: approval = approve(request)
        with spawn(at=Manager, retain=(Manager,), fresh="all"):
            if (request == "request") @ Manager:
                Manager(request) >> Reviewer(request)
            else:
                Manager(request) >> Reviewer(request)

    fork = next(node for node in _leaves(project(decision_then_spawn, Manager))
                if isinstance(node, BoundForkStmt))
    template = project_templates(decision_then_spawn)[(fork.site, "Manager")]
    assert fork.body == template.program


@fragment
def reviewer_only():
    Reviewer: approval = approve("request")


@fragment
def manager_only():
    Manager: approval = approve("request")


@pytest.mark.parametrize("retained", [None, (), (Reviewer,)])
def test_initiator_must_be_retained(retained):
    with pytest.raises(ValueError, match="initiator must be a retained child role"):
        SpawnStmt(allocated.body.body, owner=Manager, retain=retained)


@pytest.mark.parametrize("mode", ["()", "'all'", "all_except(Reviewer.request)"])
def test_initiator_retention_checked_for_every_freshness_form(mode, tmp_path):
    from zippergen.workflow_io import load_workflow_spec
    path = tmp_path / "invalid.py"
    path.write_text("from zippergen import *\n"
                    "Manager, Reviewer = Lifeline('Manager'), Lifeline('Reviewer')\n"
                    "request = Var('request', str, default='request')\n"
                    "@workflow\ndef bad():\n"
                    f"    with spawn(at=Manager, retain=(Reviewer,), fresh={mode}):\n"
                    "        Manager(request) >> Reviewer(request)\n")
    with pytest.raises(ValueError, match="initiator must be a retained child role"):
        load_workflow_spec(str(path) + ':bad')


def test_initiator_must_occur_in_child():
    with pytest.raises(ValueError, match="distinct child roles"):
        @workflow
        def invalid():
            spawn(reviewer_only, at=Manager, retain=(Manager,))


def test_empty_initiated_child_rejected_but_inherited_empty_allowed():
    with pytest.raises(ValueError, match="distinct child roles"):
        @workflow
        def invalid():
            with spawn(at=Manager, retain=(Manager,)):
                pass
    @workflow
    def inherited_empty():
        with spawn():
            pass
    assert inherited_empty.body.owner is None


def test_dispatcher_uses_trivial_retained_child_role():
    @workflow
    def dispatch():
        with spawn(at=Manager, retain=(Manager,)):
            skip(Manager)
            reviewer_only()
    code = _leaves(project(dispatch, Manager))
    assert [type(node) for node in code] == [InitSendStmt, BoundForkStmt]
    assert code[-1].body.__class__.__name__ == "SkipStmt"
    assert participation_set(dispatch.body) == frozenset({Manager})


def test_at_requires_lifeline_and_obsolete_bindings_keyword_is_rejected():
    with pytest.raises(TypeError, match="Lifeline"):
        @workflow
        def bad_owner():
            spawn(reviewer_only, at="Manager")

    with pytest.raises(TypeError, match="bindings"):
        @workflow
        def removed_binding_api():
            spawn(reviewer_only, bindings={"Reviewer": "r"})
