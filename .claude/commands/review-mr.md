Review the code changes in GitLab MR $ARGUMENTS and report findings to stdout.

Reviews the **remote MR state**, not local `HEAD`. Adhere to @CLAUDE.md.

Run each command as a separate Bash call with no shell metacharacters (no `$(...)`, pipes, `;`, `&&`/`||`, redirections, here-docs).

### 1. Identify the MR — resolve `<MR_IID>`
- Number given (e.g. `8786`) → use it.
- Diff file given → ask user for source branch, then `glab mr list --source-branch <branch>`.
- Nothing given → `git branch --show-current` (if empty, ask); then `glab mr list --source-branch <branch>`.

If `glab mr list` returns 0 MRs, stop and report. If >1, ask which IID.

### 2. Get MR metadata
1. `glab api "projects/:id/merge_requests/<MR_IID>"` — record `sha` as `<MR_HEAD_SHA>` and `source_branch`.
2. `glab api "projects/:id/merge_requests/<MR_IID>/changes"` — diff payload.

### 3. Drift check
`<source_branch>` comes from the GitLab API and is trusted (GitLab validates branch names; PETSc convention narrows further).
- `git show-ref --verify --quiet refs/heads/<source_branch>` — if non-zero, skip.
- Else `git rev-parse <source_branch>`; if it differs from `<MR_HEAD_SHA>`, warn that local and MR head diverge and recommend `/review-branch`.

### 4. Read and review the diff
- Fetch with `glab api "projects/:id/merge_requests/<MR_IID>/changes" | jq '.changes |= map(select(.new_path | endswith(".out") | not))'` to drop `.out` files (test reference output, not code). Or read a local diff file and skip `.out` hunks.
- Act as a senior software engineer. Focus on:
  - Bugs and correctness issues
  - Performance implications
  - Code quality, style, and documentation: check against conventions in @doc/developers/style.md
  - Missing error handling
- Never review `.out` file *contents*; do flag mismatches between a code change and its reference output (missing update, unjustified regeneration, orphan file).
- PETSc error model: treat `PetscCall()`, `PetscCheck()`, `SETERRQ` as terminal — don't report leaks/un-restored arrays on fatal paths. Do report bugs on non-error paths or before the error fires.
- Classify each finding: CRITICAL / HIGH / MEDIUM / Style / LOW. Don't praise the MR.
- Enumerate exhaustively: list every occurrence of each issue, not a representative example. If a pattern (e.g. missing `PetscCall`, brace-on-single-statement, hoisted-decl violation) appears in N places, report all N with file:line. Do not collapse repeats into "and similar elsewhere"; do not stop at "enough" findings. Scan every changed hunk before reporting.

Severity weights for PETSc:
- **CRITICAL / HIGH / MEDIUM** — correctness, performance, real bugs.
- **Style** — important. PETSc convention violations (clang-format, naming, idioms, CLAUDE.md anti-patterns) are real review blockers. Treat at par with MEDIUM.
- **LOW** — count, do not list. End the report with `(N LOW findings suppressed; ask to show them.)` when `N > 0`. List individual LOW items only if asked.

### 5. Verify each finding before reporting
Blocking step. For every finding at Style or above, confirm it against current code with a concrete grep/read — not from the hunk alone. Watch for "missing" claims caused by the workflow itself (e.g. the `:(exclude)*.out` filter in §4 hides reference-output files). Drop anything that does not survive verification; "Enumerate exhaustively" in §4 governs duplicates, not unverified suspicions.

### 6. Report
Per finding: severity, file:line, description, suggested fix. Order CRITICAL → HIGH → MEDIUM → Style. If nothing at or above Style is found, say so explicitly.
