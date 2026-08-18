---
name: review-branch
description: Review a local PETSc branch's changes against its target branch (origin/main or origin/release). Use when the user asks to "review this branch", "review my changes", "check what I've done before pushing", or runs `make branch-review`.
argument-hint: <branch | commit-ref | empty for HEAD>
---

Adhere to @AGENTS.md.

`SRC` is `$ARGUMENTS` if given, else `HEAD`. Reject anything that isn't a single ref matching `^[A-Za-z0-9._/@][A-Za-z0-9._/@~^-]*$`.

Resolve `DEST`:

```
MB=$(git merge-base origin/main <SRC>) && git merge-base --is-ancestor "$MB" origin/release && echo origin/release || echo origin/main
```

If `origin/main` doesn't resolve (`git rev-parse --verify -q origin/main` exits non-zero), first run `git fetch -q --no-tags origin +release:refs/remotes/origin/release +main:refs/remotes/origin/main`, then retry. Any other failure: abort and report — do not guess `DEST`.

State `DEST`, then size with `git diff --shortstat <DEST>...<SRC>`. Build `branch-review.txt`:

- `git diff <DEST>...<SRC> > branch-review-raw.diff`
- `awk '/^diff --git /{s=($0 ~ /\.out"?$/); if (s) {print $0 " [.out reference; body omitted]"; next}} s && /^(new file mode|deleted file mode|rename from|rename to|similarity index) /{print; next} !s' branch-review-raw.diff > branch-review.txt`
- Capture check: stop and report unless `grep -c '^diff --git ' branch-review-raw.diff` and `grep -c '^diff --git ' branch-review.txt` print the same non-zero number.

Then follow @../review-mr/review-procedure.md (Sections 4–6) to classify, verify, and report findings. Report to stdout only — skip Section 7; do not write `ai-review.html`.
