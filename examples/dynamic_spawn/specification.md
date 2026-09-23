# Onboarding with fresh Employee endpoints

HR receives two hires, Alice and Bob. For each arrival it starts the statically
compiled Onboarding protocol. HR retains its existing endpoint; spawn allocates
a distinct new endpoint for Employee and starts that role's local program.
The intake loop and both child HR programs share the original HR endpoint.
There is no join in the parent and no global activation barrier.

Before changing the intake variable for the next arrival, HR copies `hire`
into a child-local cell at its own fork. Each Employee endpoint starts with
uninitialized protocol fields. HR's message initializes Employee.details;
Employee's confirmation action initializes Employee.response. The response
returns explicitly to HR in the same session. HR.confirmation is also fresh
so concurrent replies do not overwrite one another's cells.

This example uses deterministic stand-ins for an arrival source and human
confirmation. Creating an endpoint creates a local execution worker, not a
person, account, credential or machine. A production connector would deliver
the question to the person identified by the received hire data.

Two arrivals must produce distinct session and Employee endpoint identities,
correctly correlated replies, and isolated hire inputs. The parent may finish
before the children. The runner returns after all sessions finish. Its return
value counts dispatched hires, not successful human approvals.

Run from the repository root:

```sh
python examples/dynamic_spawn/workflow.py
zg validate examples/dynamic_spawn/workflow.py:onboard_employees
zg show --communications examples/dynamic_spawn/workflow.py:onboard_employees
zg show --agent Employee examples/dynamic_spawn/workflow.py:onboard_employees
```

No endpoint registry or binding value is supplied by the application. The
runtime allocates and records fresh identities, installs the static role
templates, and recovers them from the durable store after a restart.
