# Operating Model

## Goal

Use this repository as the shared operational memory for:

- what is live now
- what we are practicing next
- how Sunny gets back to a known-good state
- how collaborators share work without losing reproducibility

## Branching Model

Use git branches as the default exchange mechanism for anything that needs to be shared across machines or people.

Recommended baseline:

- `main`: stable docs, stable boot scripts, stable practice code
- `feature/*`: active experiments
- `ops/*`: operational changes to boot logic, service definitions, and infrastructure scripts

This is stricter than copying files around, but it buys us something important: the machine state stops being implicit.

## Deployment Rule

If a change affects any of these, it should land in the repo:

- service start commands
- port bindings
- health checks
- environment assumptions
- reboot recovery steps
- training or serving entry points

If a change is still too unstable to commit, manual copy is fine, but it should be treated as a staging step rather than the long-term mechanism.

## Sunny Workflow

For Sunny, the intended loop is:

1. Update the repo on the machine.
2. Run the repo-owned re-init script.
3. Audit the stack from the repo.
4. If the result is good, keep the branch or merge it.
5. If the result is bad, revert or switch branches instead of guessing.

That is the core reason to move this logic into git: we get a named, reproducible machine state instead of an oral tradition.

## What The Repo Owns

The repo should own:

- boot scripts
- audit scripts
- health checks
- service inventory docs
- training and serving code
- infrastructure scripts that can be replayed

## What The Repo Does Not Own

The repo should not own:

- model weights under a host-local path such as `C:\ml-lab\models`
- Twilio credentials
- AWS credentials
- local venv caches
- mutable production data

Those stay host-local or in secret managers, but the repo should still document where they are expected to exist.

## Practical Benefit

The point is not process for process's sake.

The point is that, when Sunny reboots or a second collaborator needs to understand the system, we should be able to answer:

- what should be running
- why it should be running
- how to start it
- how to verify it
- what is experimental versus stable

without reconstructing that from memory each time.
