Review a local branch's changes against its target branch. Adhere to @CLAUDE.md.

`SRC` is `$ARGUMENTS` if given, else `HEAD`. Reject anything that isn't a single ref matching `^[A-Za-z0-9._/@][A-Za-z0-9._/@~^-]*$`.

Resolve `DEST`:

```
MB=$(git merge-base origin/main <SRC>) && git merge-base --is-ancestor "$MB" origin/release && echo origin/release || echo origin/main
```

If `origin/main` doesn't resolve (`git rev-parse --verify -q origin/main` exits non-zero), first run `git fetch -q --no-tags origin +release:refs/remotes/origin/release +main:refs/remotes/origin/main`, then retry. Any other failure: abort and report — do not guess `DEST`.

State `DEST`, then run `git diff --stat <DEST>...<SRC> -- ':(exclude)*.out'` to size the change, and `git diff <DEST>...<SRC> -- ':(exclude)*.out' > /tmp/review.diff` to capture it. Read `/tmp/review.diff` (use `offset`/`limit` if large) — do **not** re-run `git diff` per file; the captured diff already contains every file. Then follow Sections 4–6 of @.claude/commands/review-mr.md. Any options (e.g. `--stat`) must precede the revision args, not follow the pathspec.
