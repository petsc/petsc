### 4. Read and review the diff
- MR review: build a text diff file:
  - Stop and report unless `glab mr diff <MR_IID> --raw > mr-<MR_IID>-raw.diff` exits 0.
  - `awk '/^diff --git /{s=($0 ~ /\.out"?$/); if (s) {print $0 " [.out reference; body omitted]"; next}} s && /^(new file mode|deleted file mode|rename from|rename to|similarity index) /{print; next} !s' mr-<MR_IID>-raw.diff > mr-<MR_IID>-diff.txt`
  - Capture check: stop and report unless `grep -c '^diff --git ' mr-<MR_IID>-raw.diff` and `grep -c '^diff --git ' mr-<MR_IID>-diff.txt` print the same non-zero number.
- The diff file to review is `mr-<MR_IID>-diff.txt` (MR) or the existing `branch-review.txt` (local branch), never the `-raw.diff`.
- Report any file the diff file shows as `Binary files ... differ` as not covered.
- `wc -l` the diff file and review through to that last line, in parts if needed. Do **not** re-run the diff per file.
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

### 6. Compose report
- Per finding: severity, file:line, description, suggested fix. Order CRITICAL → HIGH → MEDIUM → Style. If nothing at or above Style is found, say so explicitly.
- State any path Section 4 reported as not covered.

### 7. Write report
- Always write the report (with a title) to ai-review.html! Add a footnote with claude version and model used, date, time, MR_IID, CI_PIPELINE_ID, CI_JOB_ID, when available.
