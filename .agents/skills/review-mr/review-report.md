Always write the titled report to `ai-review.html`, including when there are no findings. Add a
footnote with the agent, version, model, and effort level actually used for this review, the
current date and time, `MR_IID`, `CI_PIPELINE_ID`, and `CI_JOB_ID`. Omit unavailable values; use
`$CLAUDE_EFFORT` only when Claude performed the review; do not infer the agent or model from
installed executables or unrelated environment settings.
