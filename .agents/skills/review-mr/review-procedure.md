### 3. Read and review the diff
- Review the file named by `DIFF_FILE`.
- Report any file the diff shows as binary (`Binary files ... differ` or `GIT binary patch`) as not covered.
- Review through to line `LINES`, in parts if needed. Do **not** re-run the diff per file.
- Focus on:
  - Bugs and correctness issues
  - Performance implications
  - Code quality, style, and documentation: check against conventions in @doc/developers/style.md
  - Unchecked or swallowed error returns (e.g. calls missing `PetscCall()`)
- Never review `.out` file *contents*; do flag mismatches between a code change and its reference output (missing update, unjustified regeneration, orphan file).
- PETSc error model: treat `PetscCall()`, `PetscCheck()`, `SETERRQ` as terminal — don't report leaks/un-restored arrays on fatal paths. Do report bugs on non-error paths or before the error fires.
- Classify each finding: CRITICAL / HIGH / MEDIUM / Style / LOW.
- A clean diff producing few or no findings is a valid outcome.

Severity weights for PETSc — every finding must carry the evidence its kind of claim requires:
- **CRITICAL / HIGH / MEDIUM** — behavioral claims: correctness, performance, real bugs. State the **trigger** (concrete input or event; an MPI rank/partition condition counts) and **impact** (concrete consequence: wrong result, crash, hang, leak; a slowdown must name the affected operation). Demote to LOW when the trigger violates documented use, or the impact is only a clumsy-but-accurate message, a lost convenience, or one cheap rerun (a test or CI job, not a production solve).
- **Style** — at par with MEDIUM; a real review blocker. Two kinds:
  - Convention violation: name the broken rule — the AGENTS.md clause, `doc/developers/style.md` section, or linter check. No trigger/impact; do not invent one ("someone reads the code" carries no information). Cannot name the rule — LOW at best.
  - Factually wrong user-facing text (docstring, error message, docs): quote the text and state what the code actually does. Never LOW; higher than Style if following the text causes damage.
- **LOW** — count, do not list; list individual LOW items only if asked.

A design alternative is not a finding.

Calibration:
- MEDIUM: `VecRestoreArray()` missing on an early-return path — trigger: the `n == 0` fast path; impact: vector left locked, next `VecGetArray()` errors.
- Style: braces around a single-statement `if` — rule: AGENTS.md anti-pattern "Braces on single-statement if/else".
- Style: error message says "matrix" where it means "vector" — untrue published text, never LOW.
- LOW: error message awkward but accurate.

### 4. Verify each finding before reporting
Treat every finding at Style or above as tentative: reopen the cited code in the current working tree and confirm the finding matches it, is real — not a misread or speculation — and is actionable. Then confirm its evidence holds — the trigger and impact justify the severity and the fix neither breaks a documented use nor adds more mechanism than the impact warrants; a cited rule says what the finding claims; quoted text really contradicts the code — and downgrade to LOW when it does not. Report only findings that survive.

### 5. Compose report
- Per finding: severity, file:line, description, its required evidence (Section 3), suggested fix. List every occurrence of a confirmed issue with file:line — never a representative example or "and similar elsewhere". Order CRITICAL → HIGH → MEDIUM → Style.
- State the coverage: files and lines reviewed, and any path Section 3 reported as not covered. End with `(N LOW findings suppressed; ask to show them.)` when `N > 0`.
- The report contains the findings, the coverage, the LOW count, and what other sections explicitly say to state — nothing else: no praise, no MR summary, no design commentary. If there are no findings at or above Style, say exactly that.
