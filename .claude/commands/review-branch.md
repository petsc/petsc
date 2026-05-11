Review a local branch's changes against its target branch. Adhere to @CLAUDE.md.

`SRC` is `$ARGUMENTS` if given, else `HEAD`. Reject anything that isn't a single ref matching `^[A-Za-z0-9._/@][A-Za-z0-9._/@~^-]*$`.

Resolve target branch (`release` or `main`). Run each as a separate Bash call with no shell metacharacters (no `$(...)`, pipes, `;`, `&&`/`||`, redirections, here-docs):

1. `git fetch -q --no-tags origin +release:refs/remotes/origin/release +main:refs/remotes/origin/main`
2. `git remote get-url origin` — URL must contain `petsc/petsc`; else ask which remote.
3. `git merge-base origin/main <SRC>` — use as `<BASE_MAIN>`. (Same SHA `git diff origin/main...<SRC>` uses.)
4. `git merge-base --is-ancestor <BASE_MAIN> origin/release` — exit 0 → `DEST=origin/release`; exit 1 → `DEST=origin/main`; any other exit code → abort and report the failure, do not guess `DEST`.

State `DEST`, then `git diff <DEST>...<SRC>` and follow Sections 4–5 of @.claude/commands/review-mr.md.
