---
name: review-branch
description: Review a local PETSc branch's changes against its target branch (origin/main or origin/release). Use when the user asks to "review this branch", "review my changes", "check what I've done before pushing", or runs `make branch-review`.
argument-hint: <branch | commit-ref | empty for HEAD>
---

`SRC` is `$ARGUMENTS` if given, else `HEAD`. Reject anything that isn't a single ref matching `^[A-Za-z0-9._/@][A-Za-z0-9._/@~^-]*$`.

Run `python3 lib/petsc/bin/maint/ai_review_fetch.py branch <SRC>` (path relative to the repository root) as a single shell command with no shell metacharacters. Stop and report if it exits non-zero.

It writes `branch-review.txt`, and prints `SRC`, `SRC_SHA`, `DEST`, `SHORTSTAT`, `DIFF_FILE`, `FILES`, and `LINES`. State `DEST` and `SHORTSTAT`.

Follow @../review-mr/review-procedure.md (Sections 3–5), skipping parts marked **MR only**, to review and verify findings and compose the report. Report to stdout only.
