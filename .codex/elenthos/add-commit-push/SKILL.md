---
name: add-commit-push
description: "INVOKE THIS SKILL when the user asks to push, publish, or send the current local work upstream and the request implies commit-plus-push. Covers reviewing the diff, staging the intended changes, writing a rationale-driven commit message, committing, and pushing the current branch."
---

<overview>
Treat requests like "push the current changes", "push this work", or "commit and push" as a full commit-and-push workflow:

1. Run `make format`
2. Run `make check`
3. Clear `PLANS.md`
4. Inspect the worktree
5. Stage the intended changes
6. Write the commit message after reviewing the diff
7. Commit
8. Push the current branch

If the user asked to push the current changes as a whole, `git add .` is the default staging step.
</overview>

<workflow>
Execute the workflow in this order.

### 1. Run formatting

Run:

```
make format
```

### 2. Run validation

Run:

```
make check
```

If `make check` fails or does not complete, ask the user how to proceed before continuing.

Ignore expected `xfail` tests when evaluating whether `make check` passed.

If `make check` succeeds, continue to clear `PLANS.md`.

### 3. Clear `PLANS.md`

Delete all content in `PLANS.md` so the file is empty before continuing. Plans are working notes and should not be committed once validation has passed.

### 4. Inspect the worktree

Run:

```
git status --short
git diff --stat
git diff
```

Use the diff and the conversation history with the user to understand what changed and why.

### 5. Stage the intended changes

If the user asked to push the current changes without narrowing scope, stage everything:

```
git add .
```

If the user clearly asked to push only part of the work, stage only the relevant paths instead of using `git add .`.

### 6. Verify the staged diff

Run:

```
git diff --cached --stat
git diff --cached
```

Confirm the staged content matches the intended scope before committing.

### 7. Write the commit message from the diff

Write the commit message only after reviewing `git diff --cached`.

The message must explain why the changes were made, not just what files changed. Prefer:

- an imperative subject line focused on intent
- a short body explaining why the changes were made

Avoid vague messages like `update files`, `misc fixes`, or `wip`.

### 8. Create the commit

Create a normal commit from the staged changes. Do not amend unless the user explicitly asked for that.

### 9. Push the current branch

Push the current branch to its remote. If no upstream is configured, determine the branch name and push with upstream configuration.

### 10. Report the result

Report:

- the branch name
- the new commit hash
- whether anything was intentionally excluded
  </workflow>

<commit-message-standard>
Use this shape when it fits:

```text
<imperative subject focused on intent>

<optional body explaining why the change was needed, what behavior it improves,
or what constraint motivated it>
```

Good commit messages are diff-aware and rationale-driven.
</commit-message-standard>

<fix-staging-before-commit>
Do not attempt to commit unstaged work.

```
# WRONG
git commit -m "..."

# CORRECT for "push the current changes"
git add .
git diff --cached
git commit -m "..."
```

</fix-staging-before-commit>
