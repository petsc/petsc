---
name: review-mr
description: Review code changes in a PETSc GitLab merge request and report findings to stdout, and to ai-review.html. Use when the user asks to "review this MR", "review MR <number>", or "look at MR !N", and wants the review printed (not posted as comments).
argument-hint: <MR_IID | empty for current branch>
---

Reviews the **remote MR state**, not local `HEAD`. Adhere to @AGENTS.md.

## Identify and fetch
Follow @identify.md (Sections 1–2) to fetch the merge request, check for drift, and repeat its warnings.

## Review
Follow @review-procedure.md (Sections 3–5) to read the diff, classify findings, verify each one, compose report.

## Write report
Always write the report (with a title) to ai-review.html! Add a footnote with claude version and model used, the effort level (read from `$CLAUDE_EFFORT`), date, time, MR_IID, CI_PIPELINE_ID, CI_JOB_ID, when available.
