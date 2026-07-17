import json
from types import ModuleType

from zippergen.deployment import DeploymentField, DeploymentSpec
from zippergen.semantic import (
    read_semantic_snapshot,
    render_semantic_diff,
    semantic_diff,
    semantic_diff_models,
    semantic_snapshot,
    workflow_semantics,
)
from zippergen.syntax import (
    ActStmt,
    EffectAction,
    IfStmt,
    Lifeline,
    MsgStmt,
    PureAction,
    Var,
    VarExpr,
    Workflow,
    seq,
)


Client = Lifeline("Client")
Worker = Lifeline("Worker")
Reviewer = Lifeline("Reviewer")

request = Var("request", str)
draft = Var("draft", str)
result = Var("result", str)


def _normalize(value: str) -> str:
    return value.strip()


pure_normalize = PureAction(
    name="normalize",
    inputs=(("value", str),),
    outputs=(("normalized", str),),
    fn=_normalize,
)
effect_normalize = EffectAction(
    name="normalize",
    inputs=(("value", str),),
    outputs=(("normalized", str),),
    fn=_normalize,
)


def _approved(env) -> bool:
    return bool(env.approved)


_approved._src = "approved"  # type: ignore[attr-defined]


before_workflow = Workflow(
    name="intake",
    inputs=(("request", str, Client),),
    output_type=str,
    vars=(draft,),
    body=seq(
        MsgStmt(Client, (VarExpr(request),), Worker, (VarExpr(request),)),
        ActStmt(Worker, pure_normalize, (VarExpr(request),), (draft,)),
        MsgStmt(Worker, (VarExpr(draft),), Client, (VarExpr(result),)),
    ),
    outputs=((result, Client),),
    ns={"Client": Client, "Worker": Worker},
)

after_workflow = Workflow(
    name="intake",
    inputs=(("request", str, Client),),
    output_type=str,
    vars=(draft,),
    body=seq(
        MsgStmt(Client, (VarExpr(request),), Worker, (VarExpr(request),)),
        ActStmt(Worker, effect_normalize, (VarExpr(request),), (draft,)),
        IfStmt(
            _approved,
            Worker,
            MsgStmt(Worker, (VarExpr(draft),), Reviewer, (VarExpr(result),)),
            MsgStmt(Worker, (VarExpr(draft),), Client, (VarExpr(result),)),
        ),
    ),
    outputs=((result, Reviewer),),
    ns={"Client": Client, "Worker": Worker, "Reviewer": Reviewer},
)


before_module = ModuleType("before")
before_module.zippergen_deployment = DeploymentSpec(name="intake")
after_module = ModuleType("after")
after_module.zippergen_deployment = DeploymentSpec(
    name="intake",
    fields=(DeploymentField("review_channel", "Review channel", required=True),),
)


def test_workflow_semantics_records_context_and_effect_kind():
    model = workflow_semantics(after_workflow, after_module)

    assert model["lifelines"] == ["Client", "Worker", "Reviewer"]
    assert model["action_definitions"]["normalize"]["kind"] == "effect"
    assert model["controls"][0]["code"] == "if (approved) @ Worker"
    assert "if:Worker:approved=true" in model["messages"][1]["context"]


def test_semantic_diff_reports_protocol_effect_and_deployment_changes():
    result = semantic_diff(
        before_workflow,
        after_workflow,
        before_module,
        after_module,
    )
    changes = result["changes"]

    assert result["changed"] is True
    assert changes["lifelines"]["added"] == ["Reviewer"]
    assert changes["action_definitions"]["changed"][0]["name"] == "normalize"
    assert changes["action_definitions"]["changed"][0]["fields"]["kind"] == {
        "before": "pure",
        "after": "effect",
    }
    assert changes["controls"]["added"][0]["kind"] == "if"
    assert changes["deployment"]["fields"]["added"][0]["name"] == "review_channel"


def test_render_semantic_diff_is_source_like_and_stable():
    result = semantic_diff(before_workflow, after_workflow, before_module, after_module)
    text = render_semantic_diff(result)

    assert text.startswith("# Semantic workflow diff: intake -> intake")
    assert "+ Reviewer" in text
    assert "~ normalize" in text
    assert "kind: pure -> effect" in text
    assert "+ if (approved) @ Worker" in text
    assert "+ fields: review_channel:" in text


def test_semantic_diff_round_trips_as_json():
    result = semantic_diff(before_workflow, after_workflow, before_module, after_module)

    assert json.loads(json.dumps(result))["changed"] is True


def test_semantic_snapshot_can_be_used_as_diff_baseline():
    snapshot = json.loads(json.dumps(semantic_snapshot(before_workflow, before_module)))
    baseline = read_semantic_snapshot(snapshot)
    result = semantic_diff_models(
        baseline,
        workflow_semantics(after_workflow, after_module),
    )

    assert snapshot["schema"] == "zippergen.workflow-semantics.v1"
    assert result["changed"] is True
    assert result["changes"]["lifelines"]["added"] == ["Reviewer"]


def test_json_round_tripped_snapshot_matches_unchanged_workflow():
    snapshot = json.loads(json.dumps(semantic_snapshot(after_workflow, after_module)))

    result = semantic_diff_models(
        read_semantic_snapshot(snapshot),
        workflow_semantics(after_workflow, after_module),
    )

    assert result["changed"] is False


def test_semantic_snapshot_rejects_unknown_schema():
    try:
        read_semantic_snapshot({"schema": "future", "workflow": {"name": "x"}})
    except ValueError as exc:
        assert "unsupported semantic snapshot schema" in str(exc)
    else:
        raise AssertionError("expected invalid semantic snapshot to fail")


def test_semantic_snapshot_rejects_incomplete_model():
    try:
        read_semantic_snapshot({
            "schema": "zippergen.workflow-semantics.v1",
            "workflow": {"name": "x"},
        })
    except ValueError as exc:
        assert "workflow model is incomplete" in str(exc)
    else:
        raise AssertionError("expected incomplete semantic snapshot to fail")


def test_semantic_diff_detects_rename_and_statement_reordering():
    renamed_reordered = Workflow(
        name="renamed_intake",
        inputs=before_workflow.inputs,
        output_type=str,
        vars=(draft,),
        body=seq(
            ActStmt(Worker, pure_normalize, (VarExpr(request),), (draft,)),
            MsgStmt(Client, (VarExpr(request),), Worker, (VarExpr(request),)),
            MsgStmt(Worker, (VarExpr(draft),), Client, (VarExpr(result),)),
        ),
        outputs=before_workflow.outputs,
        ns=before_workflow.ns,
    )

    diff_result = semantic_diff(before_workflow, renamed_reordered)

    assert diff_result["changed"] is True
    assert diff_result["changes"]["name"] == {
        "before": "intake",
        "after": "renamed_intake",
    }
    assert diff_result["changes"]["messages"] == {"added": [], "removed": []}
    assert diff_result["changes"]["action_sites"] == {"added": [], "removed": []}
    assert diff_result["changes"]["protocol"]
    text = render_semantic_diff(diff_result)
    assert "~ intake -> renamed_intake" in text
    assert "~ execution order or control structure changed" in text
