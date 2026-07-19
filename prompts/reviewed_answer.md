# Reviewed-answer tutorial workflow

Create `workflows/reviewed_answer.py` with a workflow named
`reviewed_answer`.

## Participants and behavior

- Requester initially owns `request` and `max_retries`.
- Writer uses an LLM to draft and revise an answer.
- Reviewer uses a separate LLM to assess every draft.
- After the automated assessment, a human approves or rejects the draft.
- A rejection sends the draft back to Writer while retries remain.
- Never return an unapproved draft. Return an explicit failure after retry
  exhaustion.

## Development and deployment requirements

- Declare `mock` as the default model.
- Declare `request` as a deployment input defaulting to
  `Explain why the sky is blue in two sentences.`
- Declare `max_retries` as an integer deployment input defaulting to `2`.
- Add conditional OpenAI and Anthropic API-key fields.
- Include the generated workflow file in the deployment bundle.
- Add focused tests in `tests/test_reviewed_answer.py`.
- Do not deploy or start a service.
