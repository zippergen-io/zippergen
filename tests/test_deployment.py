import json

import pytest

from zippergen.deployment import (
    DeploymentField,
    DeploymentSpec,
    normalize_deployment_spec,
)
from zippergen.serve import _collect_deployment_fields


def test_plain_mapping_declaration_normalizes_to_typed_jsonable_spec():
    spec = normalize_deployment_spec({
        "name": "generated-workflow",
        "fields": [
            {
                "name": "api_key",
                "prompt": "API key",
                "target": "env",
                "env": "GENERATED_API_KEY",
                "secret": True,
                "required": True,
            }
        ],
        "packages": [
            {"requirement": "generated-client>=1", "import_name": "generated_client"}
        ],
        "setup": [
            {
                "name": "authorize",
                "description": "Authorize the generated client",
                "command": ["{python}", "setup_client.py"],
            }
        ],
        "files": ["workflow.py", "setup_client.py"],
    })

    assert isinstance(spec, DeploymentSpec)
    assert isinstance(spec.fields[0], DeploymentField)
    assert spec.fields[0].target_name == "GENERATED_API_KEY"
    assert spec.packages[0].import_name == "generated_client"
    assert spec.setup[0].command == ("{python}", "setup_client.py")
    assert json.loads(json.dumps(spec.as_dict()))["files"] == ["workflow.py", "setup_client.py"]


def test_secret_field_must_target_environment():
    with pytest.raises(ValueError, match="must target env"):
        DeploymentField("token", "Token", secret=True)


def test_field_default_can_reference_an_earlier_field():
    spec = DeploymentSpec(fields=(
        DeploymentField("recipient", "Recipient", required=True),
        DeploymentField(
            "query",
            "Query",
            target="env",
            env="DEMO_QUERY",
            default="to:{recipient}",
        ),
    ))

    values, _secrets = _collect_deployment_fields(
        spec,
        {},
        overrides={"recipient": "calls@example.com"},
        interactive=False,
    )

    assert values["query"] == "to:calls@example.com"


def test_conditional_secret_is_enabled_by_a_per_lifeline_model():
    spec = DeploymentSpec(fields=(
        DeploymentField("llm", "Default LLM", target="llm", default="mock"),
        DeploymentField(
            "openai_api_key",
            "OpenAI API key",
            target="env",
            env="OPENAI_API_KEY",
            secret=True,
            required=True,
            when="llm",
            when_values=("openai*",),
        ),
    ))

    values, secrets = _collect_deployment_fields(
        spec,
        {
            "llm": "mock",
            "llms": {"Writer": "openai:gpt-4o-mini"},
        },
        overrides={"openai_api_key": "deployment-secret"},
        interactive=False,
    )

    assert values["openai_api_key"] == "deployment-secret"
    assert secrets == {"OPENAI_API_KEY": "deployment-secret"}
