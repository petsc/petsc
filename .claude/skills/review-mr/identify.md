Run each command in this file as a separate shell command with no shell metacharacters (no `$(...)`, pipes, `;`, `&&`/`||`, here-docs).

### 1. Identify the MR — resolve `<MR_IID>`
Reject any `<MR_IID>` that isn't `^[0-9]+$`.
- Number given (e.g. `8786`) → use it.
- Nothing given → `git branch --show-current` (if empty, ask); then `glab mr list --source-branch <branch>`.

If `glab mr list` returns 0 MRs, stop and report. If >1, ask which IID.

### 2. Get MR metadata
1. `glab api "projects/:id/merge_requests/<MR_IID>" > mr-<MR_IID>-meta.json`
2. Stop and report unless both `test -s mr-<MR_IID>-meta.json` and `jq -e '.iid and .sha and .source_branch and .diff_refs.base_sha and .diff_refs.head_sha and .diff_refs.start_sha' mr-<MR_IID>-meta.json` succeed.
3. Read `sha` (as `<MR_HEAD_SHA>`), `source_branch`, and `diff_refs` (`base_sha`, `head_sha`, `start_sha`) from that file with `jq`. The diff is fetched in @review-procedure.md Section 4.

### 3. Drift check
`<source_branch>` comes from the GitLab API and is trusted (GitLab validates branch names; PETSc convention narrows further).
- `git show-ref --verify --quiet refs/heads/<source_branch>` — if non-zero, skip.
- Else `git rev-parse <source_branch>`; if it differs from `<MR_HEAD_SHA>`, warn that local and MR head diverge and recommend `/review-branch`.
