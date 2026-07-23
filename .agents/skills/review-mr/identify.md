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
