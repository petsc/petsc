# pde-sim binding for petscagent-bench

Exposes the pde-sim pipeline as the **Purple agent** (agent-under-test) in
[petscagent-bench](https://github.com/petsc/petscagent-bench), so pde-sim can be
scored by the same gates/metrics/quality evaluators as any other code-generation
agent — **without modifying any bench source**.

## How it works

The bench Green agent is decoupled from the Purple *implementation*: it reads the
purple **URL** from its task message and talks A2A/HTTP only. This binding is a
standalone A2A server implementing the Purple contract:

1. Green sends a plain-text `problem_description`.
2. `pdesim_purple.py` drives pde-sim non-interactively via `claude -p` in
   **autonomous** mode, writing artifacts under a per-request study dir.
3. It reads the pipeline's `results-manifest.json`, then returns:
   - a `TextPart`: `Code generation successful` / `nsize:` / `cli_args:`
     (`nsize` = the chosen run's `mpi_ranks`; `cli_args` = its `command` with the
     `mpiexec -n N ./exe` prefix stripped),
   - a `FilePart` per generated source (main `.c` first, plus `.cu` /
     `*.kokkos.cxx` dependencies). The makefile is intentionally **not** sent —
     the bench's MCP server builds its own.
4. Green uploads the source via MCP, compiles, runs with `nsize`/`cli_args`, and
   scores it.

## Prerequisites

- The `claude` CLI on `PATH` (or set `PDESIM_CLAUDE_BIN`), logged in, able to run
  the `/pde-sim` pipeline in this repo.
- A working PETSc build reachable via `PETSC_DIR` / `PETSC_ARCH` in the
  environment (pde-sim compiles/verifies; the bench independently recompiles).
- The A2A server deps (`a2a-sdk[http-server]`, `uvicorn`). Simplest is to run
  this file from petscagent-bench's own `uv` env.

## Run it (no bench code changes)

Use the bench's "separate components" path — **not** `main.py launch`, which
hardcodes the built-in purple.

Export the needed environment first (the MCP server and the pde-sim pipeline need
`PETSC_DIR`/`PETSC_ARCH`; the Green judge needs its LLM key — `ARGO_API_KEY` for
Argo). `uv run` does not read the bench's `.env`, so export them in the shells:

```bash
export PETSC_DIR=/path/to/petsc PETSC_ARCH=your-arch ARGO_API_KEY=your_anl_username

# 1) Green agent (from the bench repo/venv) — :9001
cd ~/petscagent-bench && uv run src/green_agent/server.py &

# 2) PETSc compile/run MCP server — :8080
#    petsc_mcp_servers ships as a bench dependency (petsc-ai-servers-clients),
#    so its module is importable from the bench env. main() defaults to
#    streamable-http on :8080 with the /mcp endpoint:
cd ~/petscagent-bench && \
  uv run python -c "from petsc_compile_run_mcp_server import main; main()" &

# 3) This binding as the Purple agent — :9002
#    Run from the bench env so a2a-sdk/uvicorn are available:
cd ~/petscagent-bench && \
  uv run python ~/petsc/.agents/pde-sim/bindings/petscagent-bench/pdesim_purple.py &

# 4) Trigger the run, pointing Green at this binding's URL
cd ~/petscagent-bench && uv run src/client_cli.py \
  --green-url http://localhost:9001 \
  --purple-url http://localhost:9002 \
  --mcp-server-url http://localhost:8080/mcp
```

Results land in the bench's `output/` as usual. The bench's
`config/purple_agent_config.yaml` is **unused** here (it configures the built-in
purple's LLM); the Green config still applies.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `PDESIM_CLAUDE_BIN` | `claude` | Claude CLI binary |
| `PDESIM_REPO_DIR` | repo root (inferred) | cwd for the pipeline |
| `PDESIM_ARTIFACTS_DIR` | `<repo>/artifacts` | where studies/manifests are searched |
| `PDESIM_TIMEOUT` | `2700` | per-problem seconds (keep < bench's 3000s A2A read timeout) |
| `PDESIM_PERMISSION_FLAG` | `--permission-mode bypassPermissions` | Claude CLI permission flag |
| `PDESIM_CLAUDE_MODEL` | *(unset)* | optional `--model` override |
| `PDESIM_CLAUDE_EXTRA_ARGS` | *(unset)* | extra CLI args (shell-split) |
| `PDESIM_MAX_NSIZE` | `64` | cap on returned `nsize` (bench rejects larger) |

## Known limitations / things to watch

- **Grader mismatch is the main risk.** The bench's `numerical_accuracy` metric
  reads the *trailing N numeric lines* of stdout (N = length of the problem's
  hidden `expected_output`) and compares them to that reference; only
  `problem_description` is given to the agent — `test_cases` (`args` /
  `expected_output`) are not. The binding's prompt therefore tells pde-sim to
  emit **only** the output the description asks for (e.g. the final solution via
  `VecView()`) and to keep extra numeric diagnostics off stdout. A correct
  solution can still miss the metric if the description's output format is
  ambiguous; remaining cases need per-problem handling in the bench evaluators.
- **Run selection is a heuristic.** The binding sends the first successful run's
  `command`/`mpi_ranks`. If a problem needs a different run, adjust `_pick_run`.
- **cli_args are passed verbatim** after stripping the launcher/exe prefix,
  including MMS-specific flags (e.g. `-mms_levels 4`, `-mms_csv convergence.csv`);
  the latter just writes a file in the bench work dir.
- **Cost/time.** A full pde-sim pipeline (multi-agent, MMS studies, retries) uses
  far more tokens/time per problem than a single-shot purple agent, and that
  spend is **not** reflected in the bench's token counts.
- **`bypassPermissions` runs arbitrary Bash** — only run against a sandboxed
  PETSc build.
