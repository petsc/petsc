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
After generating the review, treat every finding at Style or above as tentative. For each one: reopen the cited code in the current working tree and confirm it matches what the finding describes; reread that code and confirm the issue is real, not a misread or speculation; and confirm it is actionable. Drop findings that fail any check; report only those that survive.

### 6. Report
- Per finding: severity, file:line, description, suggested fix. Order CRITICAL → HIGH → MEDIUM → Style. If nothing at or above Style is found, say so explicitly.
- If CI_PIPELINE_ID is available: add a footnote with claude version and model used, date, time, MR_IID, CI_PIPELINE_ID, CI_JOB_ID; and write the report as a standalone HTML document (with a title) to ai-review.html.
