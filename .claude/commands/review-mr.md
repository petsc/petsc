Review the code changes in GitLab MR $ARGUMENTS and report findings to stdout.

## Steps

### 1. Identify the MR
- If an MR number is given (e.g. `8786`), use it directly.
- If a diff file path is given, read it and find the MR via `glab mr list --source-branch <branch>`.
- If nothing is given, use the current branch: `glab mr list --source-branch $(git branch --show-current)`.

### 2. Get MR metadata
Run these to collect the SHAs and file list:
```
glab api "projects/:id/merge_requests/<MR_IID>/versions" # get base_sha, head_sha, start_sha from latest version
glab api "projects/:id/merge_requests/<MR_IID>/changes"   # get changed file paths
```

### 3. Read and review the diff
- Use `glab api "projects/:id/merge_requests/<MR_IID>/changes"` or read a local diff file.
- Act as a senior software engineer. Focus on:
  - Bugs and correctness issues
  - Performance implications
  - Code quality, style, and documentation: check against conventions in `doc/developers/style.md`
  - Missing error handling
- Classify each finding by severity: CRITICAL, HIGH, MEDIUM, LOW, or Style/Nit.

### 4. Report findings
For each finding, print to stdout:
- **Severity** (CRITICAL/HIGH/MEDIUM/LOW/Style)
- **File and line number**
- **Description** of the issue
- **Suggested fix** (if applicable)
