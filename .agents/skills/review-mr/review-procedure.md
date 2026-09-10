### 3. Read and review the diff

- Read `DIFF_FILE` through line `LINES`; do not regenerate the diff per file. Report binary files as not covered.
- Review correctness, performance, source style, documentation, and error handling (for example calls missing `PetscCall()`) against `AGENTS.md`, its applicable convention references, and `doc/developers/style.md`.
- Never review `.out` contents. Flag missing, unjustified, or orphaned expected-output updates.
- Treat `PetscCall()`, `PetscCheck()`, and `SETERRQ` as terminal on error. Report resource errors on normal paths or before an error fires, not missing cleanup on fatal paths.

#### MR only: Revision and temporary CI commits

- Before reviewing, check that metadata `sha` equals `diff_refs.head_sha`. If they differ, refetch diff and metadata together; stop and report if the mismatch persists.
- Identify temporary `runjobs.py` changes from the MR commit list and patches: subjects start with `DRAFT: CI: Temporary commit, remove before merge!`, and `.gitlab-ci.yml` contains the generated-command header. A mention of `runjobs` is insufficient.
- Exclude only changes attributable to those commits from findings, LOW counts, and comments. Record each excluded SHA and title. Review independent CI changes and `runjobs.py` itself; report unclear attribution as a coverage limitation. Preserve `DIFF_FILE`, the reviewed head SHA, and posting anchors.

#### Classify findings

| Severity | Required evidence |
| --- | --- |
| CRITICAL / HIGH / MEDIUM | A concrete trigger (an input, event, or MPI rank/partition condition) and a concrete impact: wrong result, crash, hang, leak, or slowdown of a named operation. Discard a trigger outside documented use unless the change breaks a guarantee documented for that case. |
| Style (a review blocker, alongside MEDIUM) | For a convention violation, name the `AGENTS.md` clause, convention reference, style-guide section, or linter check; do not invent a trigger or impact. For factually wrong user-facing text, quote it and state what the code does; use a higher severity if following the text causes damage. |
| LOW | Use LOW for confirmed issues whose impact is only awkward but accurate wording, lost convenience, or one cheap test/CI rerun, not a production solve. Count these; list them only when asked. |

A design alternative is not a finding. Factually wrong user-facing text is never LOW.
For example, a missing `VecRestoreArray()` on an `n == 0` normal return is MEDIUM if it leaves
an array locked and the next access fails. Braces around a single-statement branch are Style
under `AGENTS.md`'s brace rule. No findings is a valid result.

### 4. Verify each finding before reporting

Report only verified, actionable findings, including in the LOW count. Use `DIFF_FILE` and any
needed source, callees, macros, or declarations at `SRC_SHA` or `MR_HEAD_SHA`. Working-tree
source is evidence only when it matches that revision and has no relevant uncommitted changes.
If CodeGraph indexes another checkout, use it to locate relationships, then read the reviewed
revision; this is necessary verification, not a duplicate read.

Discard candidates that verification disproves or that lack the evidence Section 3 requires.
If necessary context is unavailable, record the path and missing evidence as a coverage
limitation. Check that the severity meets Section 3 and that the fix preserves documented use
and is proportionate to the impact.

### 5. Compose report

- For each finding, explain the problem, required evidence, and suggested fix in complete prose. Include severity and every occurrence's `file:line` within the reviewed diff. Order CRITICAL → HIGH → MEDIUM → Style.
- When a finding has a small, unambiguous fix, follow the explanation with a fenced `diff` block showing the minimal patch against the reviewed revision. Include necessary companion changes, or explicitly identify any omitted ones. Distinguish verification of the finding from testing the proposed patch; label untested patches. If the fix requires an unresolved design decision, explain the required change without inventing a patch.
- State files and lines reviewed, uncovered paths, and verification limitations, even with no findings.
- **MR only:** include excluded temporary CI commits and any attribution limitations from Section 3, plus fetch warnings required by `identify.md`.
- Include only findings, coverage, LOW counts, and information explicitly required by this procedure or the calling skill. Exclude praise, MR summaries, and design commentary. If there are no findings at or above Style, say exactly that.
- End with `(N LOW findings suppressed; ask to show them.)` when `N > 0`.
