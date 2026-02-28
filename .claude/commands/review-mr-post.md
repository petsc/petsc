Review the code changes in GitLab MR $ARGUMENTS and post inline comments with apply-able suggestions.

## Steps

### 1–3. Review the MR
Follow steps 1–3 from `.claude/commands/review-mr.md` to identify the MR, collect metadata, and review the diff.

### 4. Filter findings
Only post findings that have a **concrete, actionable fix** (a code change the author can apply). Do NOT post:
- Informational or observational notes ("just noting...", "no issue, but...")
- Findings that acknowledge correctness but flag theoretical fragility
- Style nits with no specific suggested change

Each posted comment opens a discussion thread the author must resolve — avoid noise.

### 5. Post inline comments as DiffNotes
Use the GitLab Discussions API with JSON input to create inline DiffNote comments.

**IMPORTANT:** Use `--input -` with `-H "Content-Type: application/json"` — the `-f` flag with bracket notation does NOT work for nested `position` fields.

For each comment, use this Python pattern:
```python
import json, subprocess

payload = {
    "body": "<comment body with optional suggestion block>",
    "position": {
        "position_type": "text",
        "base_sha": "<BASE_SHA>",
        "head_sha": "<HEAD_SHA>",
        "start_sha": "<START_SHA>",
        "new_path": "<file_path>",
        "old_path": "<file_path>",
        "new_line": <line_number_in_new_file>
    }
}
p = subprocess.run(
    ["glab", "api", "projects/:id/merge_requests/<MR_IID>/discussions",
     "-X", "POST", "--input", "-", "-H", "Content-Type: application/json"],
    input=json.dumps(payload), capture_output=True, text=True, check=True
)
assert '"type":"DiffNote"' in p.stdout, f"Unexpected response: {p.stdout}"
```

### 6. Use GitLab suggestion blocks for concrete fixes
When a comment has a specific code fix, include a suggestion block in the body so the author can click "Apply suggestion":

````
```suggestion:-0+0
corrected line here
```
````

- `-0+0` means replace just the target line (the `new_line` in the position). Use `-N+M` to expand the range to N lines before and M lines after the target line.
- The suggestion body **replaces the entire selected range**. You MUST reproduce every line in the range, not just the changed ones. For example, `suggestion:-2+0` selects 3 lines (2 before + the target); the body must contain all 3 lines (with your edits applied). Omitting unchanged lines will delete them.
- To insert a new line after the target, use `suggestion:-0+0` and include both the original target line and the new line.
- To delete a line, use an empty suggestion block.
- Only use suggestions for concrete fixes. Use plain comments for design/architectural feedback.

### 7. Line number mapping
- For **new files**: `new_line` = the line number in the file itself.
- For **modified files**: `new_line` = the line number in the new version of the file. Parse the `@@` hunk headers to map correctly.

### 8. Verify
After posting, confirm each response has `"type":"DiffNote"` to ensure comments appear inline on the Changes tab.
