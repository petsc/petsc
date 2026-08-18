---
name: review-mr
description: Review code changes in a PETSc GitLab merge request and report findings to stdout, and to ai-review.html. Use when the user asks to "review this MR", "review MR <number>", or "look at MR !N", and wants the review printed (not posted as comments).
argument-hint: <MR_IID | empty for current branch>
---

Reviews the **remote MR state**, not local `HEAD`. Adhere to @CLAUDE.md.

## Identify and fetch
Follow @identify.md (Sections 1–3) to resolve `<MR_IID>`, fetch metadata, and check for local-vs-remote drift.

## Review
Follow @review-procedure.md (Sections 4–7) to read the diff, classify findings, verify each one, compose and write report.

## Verify report
Verify that the ai-review.html report was written, as instructed in Section 7.
