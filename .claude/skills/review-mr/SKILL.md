---
name: review-mr
description: Review code changes in a PETSc GitLab merge request and report findings to stdout. Use when the user asks to "review this MR", "review MR <number>", "look at MR !N", or provides a merge_requests URL/IID, and wants the review printed (not posted as comments).
argument-hint: <MR_IID | diff-file | empty for current branch>
---

Reviews the **remote MR state**, not local `HEAD`. Adhere to @CLAUDE.md.

## Identify and fetch
Follow @identify.md (Sections 1–3) to resolve `<MR_IID>`, fetch metadata, and check for local-vs-remote drift.

## Review
Follow @review-procedure.md (Sections 4–6) to read the diff, classify findings, verify each one, and report.
