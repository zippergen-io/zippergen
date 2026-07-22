# Reviewed Answer

Produce a helpful answer to a request, subject to automated review and explicit
human approval.

## Participants and behavior

- Requester initially owns `request` and `max_retries`.
- Writer uses an LLM to draft and revise an answer.
- Reviewer uses a separate LLM to assess every draft.
- After the automated assessment, a human approves or rejects the draft.
- A rejection sends the draft back to Writer while retries remain.
- Never return an unapproved draft. Return an explicit failure after retry
  exhaustion.

## Inputs and operation

- The application accepts a `request` input, defaulting to
  `Explain why the sky is blue in two sentences.`
- It accepts an integer `max_retries` input, defaulting to `2`.
- Use the deterministic mock model by default so the application can be tried
  without a paid model account.
- A deployment may instead use OpenAI or Anthropic. Request the corresponding
  API key only when that provider is selected.
