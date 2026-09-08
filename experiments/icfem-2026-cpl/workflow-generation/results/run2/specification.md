# Coin result along a chain

Implement a ZipperGen workflow with four participants named `L0`, `L1`, `L2`,
and `L3`.

The workflow receives two string inputs at `L0`: `outcome` and `token`.
`outcome` is the result of a private coin toss and is either `heads` or `tails`.
The other input is an opaque token.

The application-message architecture consists exactly of the following chain:

1. `L0` sends `token` to `L1`.
2. `L1` forwards `token` to `L2`.
3. `L2` forwards `token` to `L3`.

The token does not contain the coin outcome. After receiving it, `L3` itself
must decide whether the outcome stored at `L0` is `heads` or `tails`; the owner
of this decision must be `L3`. It then returns the corresponding string.

Keep the participants, inputs, and application-message architecture as stated.
Supply small deterministic local action bodies where needed so that the project
is self-contained and can be validated. Name the workflow `coin_chain`.
