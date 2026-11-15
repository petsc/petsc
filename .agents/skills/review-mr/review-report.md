Always write the titled report to `ai-review.html`, including when there are no findings. Give
the Commits section from @review-procedure.md Section 5 its own heading, and show each commit's
Action as a colored label: green for `ok`, amber for `fix message`, and red for the rest. Put
each finding's patch in its own `<pre class="patch">` right after that finding, as an
HTML-escaped diff that `git apply --check` accepts, with no other markup inside. Copy
@patch.html verbatim into the report's `<head>`. Its script colors the lines and adds a Copy
button to each patch. Add a
footnote with the agent, version, model, and effort level actually used for this review, the
current date and time, `MR_IID`, `CI_PIPELINE_ID`, and `CI_JOB_ID`. Omit unavailable values; use
`$CLAUDE_EFFORT` only when Claude performed the review; do not infer the agent or model from
installed executables or unrelated environment settings.
