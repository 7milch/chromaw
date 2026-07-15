# Architecture Decision Records

## ADR-001: Reject startup on multi-instance lock conflict (no read-only degrade)

- Status: Accepted
- Date: 2026-07-15
- Related: technical-spec.md §9.3, roadmap M2-8, src/chromaw/lock.py

### Context

technical-spec §9.3 lists two options for what happens when `chromaw` starts
against a `chroma_path` that another live `chromaw` process already holds
the `.chromaw/lock` for: (a) start anyway in read-only mode, or (b) refuse
to start. The spec presents these as examples, not a final decision.

### Decision

`ChromawLock.acquire()` raises `LockHeldError` and the CLI exits with a
non-zero status when another live process holds the lock, for **both**
read-only and `--write` sessions. There is no automatic read-only
degradation.

### Rationale

- **Simplicity and no surprising behavior.** A CLI invocation that silently
  changes what mode it runs in based on another process's state is
  confusing: the user asked for `chromaw ./chroma --write` and would
  instead get a read-only UI with no write controls, discovering the
  discrepancy only once they try to edit something.
- **Avoids needing to know the other holder's mode.** To decide whether
  degrading to read-only is even meaningful, we'd need to know whether the
  existing holder is itself running in write mode (if it's already
  read-only, there's arguably no conflict at all for a second read-only
  viewer). That requires the lock payload to carry and trust the holder's
  mode, and requires resolving what happens when *that* changes over the
  holder's lifetime (e.g. it wasn't in write mode at lock time but the user
  had two tabs and enabled write later) -- complexity with little payoff.
- **Consistent with safe-by-default.** Refusing outright is the simpler
  failure mode to reason about and matches the project's general bias
  toward explicit, boring behavior over automatic mode switching
  (CLAUDE.md "safe-by-default").
- The error message tells the user the holder's pid and start time so they
  can decide to stop that process, or delete a stale lock file manually if
  they believe it's wrong.

### Consequences

- Running a second `chromaw` (read-only or write) against a directory that
  already has a live instance always fails fast with a clear error,
  instead of silently downgrading.
- If a future need arises for concurrent read-only viewers alongside a
  write session, it should be a deliberate, separately-specified feature
  (e.g. a distinct "viewer" lock class) rather than an implicit fallback
  from a failed write-mode acquire.
