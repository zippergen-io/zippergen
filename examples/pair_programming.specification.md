# Pair programming

Human owns the task and every continuation decision. Implementer may edit the
workspace. Reviewer independently inspects the proposal with read-only access.
Each role keeps its own assistant session within a run.

Human chooses Ship it, Another round, or Abandon. Another round sends the full
review and an optional human note to Implementer. Human notes amend the task
for subsequent rounds and must reach both Implementer and Reviewer. Amendments
accumulate in order. A later amendment takes precedence where it changes an
earlier requirement. Empty notes leave the task unchanged.

Automatic mode uses the review verdict and supplies no human note. The round
limit bounds continuation. Shipping and abandonment end the workflow without
committing, publishing, or reverting workspace edits.
