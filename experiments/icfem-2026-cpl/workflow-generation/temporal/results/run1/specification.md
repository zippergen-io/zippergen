# Persistent check result along a chain

Implement a ZipperGen workflow with four participants named L0, L1, L2,
and L3.

The workflow receives four string inputs at L0: first_status,
second_status, third_status, and token. Each status is passed, checking, or
failed. At L0, record the three supplied statuses in order as three consecutive
updates of the same application variable, named status. Each update must be a
separate local action event.

The application-message architecture consists exactly of the following chain:

1. L0 sends token to L1.
2. L1 forwards token to L2.
3. L2 forwards token to L3.

The application messages carry only the opaque token. After receiving it, L3
itself decides whether L0 has a valid pass. A passed update establishes a valid
pass, later checking updates preserve it, and a failed update cancels it. A
later passed update establishes a valid pass again.

The owner of this decision must be L3. The decision is based only on the
successive values of the recorded status variable. The three status inputs
serve only to drive these updates and must not be referenced when making the
decision. Do not maintain another variable that summarizes the status history.
Return accept when the pass is valid and reject otherwise.

Keep the participants, inputs, status-update sequence, and application-message
architecture as stated. Supply small deterministic local action bodies so that
the project is self-contained and can be validated. Name the workflow
status_chain.
