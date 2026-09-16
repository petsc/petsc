# pde-sim — a multi-agent system for PETSc-based PDE simulation

A multi-agent system that takes a **text description of a physical phenomenon**
and drives it through modeling → discretization → code generation →
verification → simulation → analysis, producing verified and analyzed numerical
solutions — visualized too when the user asks to see them.

The design is split into a framework-neutral **core** (contracts, agent role
prompts, domain-knowledge skills, example components, tests) and a thin per-tool
**binding**. Inside PETSc the core lives under `.agents/`: the domain **skills**
in `.agents/pde-sim/skills/` (deliberately **not** in the auto-discovered
`.agents/skills/`, so they never load until the pipeline is invoked), the four
agent role prompts in `.agents/pde-sim/agents/`, and the contracts, components, tests, and
docs under `.agents/pde-sim/` (this directory). The only binding built so far
is for **Claude Code**; see [Running under Claude Code](#running-under-claude-code)
and [Portability](#portability).

> Paths written as `contracts/…`, `components/…`, `skills/…`, `tests/…`, `docs/…`
> below are relative to this `.agents/pde-sim/` directory. Agent role prompts
> are referenced by their repo-root paths under `.agents/pde-sim/agents/`.

## Architecture at a glance

```
                               user: "simulate X"
                                        │
                                        ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │  ORCHESTRATOR  (main session · /pde-sim → orchestration brief) │
       │  only it dispatches subagents · every handoff flows through it  │
       └─────────────────────────────────────────────────────────────────┘
                                        │
                        dispatches each agent in pipeline
                        order, routing each output onward
                                        │
                                        ▼
  reads                               agent             produces
  ───────────────────────────────────────────────────────────────────────────────
  phenomenon text          ─▶ [ pde-modeling ]       ─▶ problem-spec.json
  problem-spec.json        ─▶ [ numerical-analysis ] ─▶ numerical-plan.json
  geometry/grid/fields     ─▶ [ visualization ]      ─▶ vis-spec.json   (opt-in)
  numerical-plan.json
  + vis-spec in-situ items ─▶ [ code-generation ]    ─▶ sim code incl. IN-SITU viz;
                                                        builds · runs · MMS-verifies;
                                                        results-manifest.json
  results-manifest.json    ─▶ [ numerical-analysis ] ─▶ numerical-assessment.json
  results-manifest.json    ─▶ [ visualization ]      ─▶ generates viz code (post-processors) & RUNS it;
                              (opt-in)                   analysis-report.json

  revision loop: orchestrator routes assessment / report back ─▶ [ numerical-analysis ] or [ pde-modeling ]

  Visualization code has two homes:
    • in-situ   — compiled INTO the simulation binary        → code-generation (per vis-spec in-situ items)
    • post-hoc  — standalone scripts run after the simulation → visualization   (→ analysis-report.json)
```

Three kinds of building block:

| Kind | Location | What it is |
|------|----------|------------|
| **Agents** (roles) | `.agents/pde-sim/agents/*.md` | Role prompts dispatched by the orchestrator as `general-purpose` subagents (own context; model inherited from the session). |
| **Skills** (knowledge) | `.agents/pde-sim/skills/*/SKILL.md` | On-demand expertise, loaded by path by whichever agent needs it. Two are **shared**. |
| **Contracts** (interfaces) | `contracts/*.schema.json` | JSON Schemas for every inter-agent handoff. The robustness backbone. |

### Why the orchestrator is the *main session*, not a subagent
Only the top-level (main) session can dispatch subagents (subagents cannot spawn
subagents). So the orchestrator is realized as the `orchestration` **skill** that
the main session loads; the four specialists are subagents it dispatches. All
coordination flows through the orchestrator — specialists never call each other.

## The agents

| Agent | Consumes | Produces | Loads skills |
|-------|----------|----------|--------------|
| `pde-modeling` | phenomenon text | `problem-spec.json` | `pde-formulation` |
| `numerical-analysis` | `problem-spec.json`; `results-manifest.json` | `numerical-plan.json`; `numerical-assessment.json` | `numerical-methods`, `petsc-solvers` |
| `code-generation` | `numerical-plan.json` (+ in-situ items of `vis-spec.json`) | simulation code incl. output + `results-manifest.json` | `petsc-codegen`, `petsc-solvers`, `pde-visualization` |
| `visualization` (opt-in) | geometry/grid/fields; `results-manifest.json` | `vis-spec.json`; standalone post-processors it runs; `analysis-report.json` | `pde-visualization` |

`petsc-solvers` and `pde-visualization` are **shared** skills — this is why
skills are named by knowledge domain, not by owning agent.

## The contracts (handoffs)

| Schema | From → To | Purpose |
|--------|-----------|---------|
| `problem-spec.schema.json` | pde-modeling → numerical-analysis | continuous math model (equations, geometry, BC/IC, parameters, scales) |
| `numerical-plan.schema.json` | numerical-analysis → code-generation | grid class, discretization, coefficient functions, solver stack, MMS, verification plan |
| `vis-spec.schema.json` | visualization → code-generation (in-situ items) | visualizations/analytics to produce, each tagged `in_situ` or `post_hoc` |
| `results-manifest.schema.json` | code-generation → orchestrator | inventory of build + runs + output file paths + convergence data |
| `numerical-assessment.schema.json` | numerical-analysis → orchestrator | convergence/conservation verdicts + recommended revisions |
| `analysis-report.schema.json` | visualization → orchestrator | artifacts produced + **factual** diagnostics (no interpretation) |

Large data (solution fields) is **referenced by path**, never inlined. At runtime,
per-study artifacts are written under `artifacts/<study-id>/` (a scratch directory,
git-ignored). Completed worked studies are checked in under `examples/<study-id>/`,
and solvers promoted from them under `components/<component>/`.

## Pipeline stages (orchestrator)

1. **Model** — pde-modeling: text → `problem-spec.json`.
2. **Discretize** — numerical-analysis: → `numerical-plan.json` (incl. MMS +
   expected convergence rates). If visualization was requested, optionally
   visualization → `vis-spec.json`.
3. **Generate & verify** — code-generation: write code, build, run MMS/
   convergence study → `results-manifest.json`. Back up a stage on failure.
4. **Run studies** — orchestrator plans a campaign; code-generation executes;
   the manifest always goes to numerical-analysis (interpret), and to
   visualization (render + diagnose) **only when visualization was requested**.
5. **Deliver** — verified and analyzed solutions; visualized when the user asked.

Then **escalate**: more physics, parametric sweeps, or design optimization.

> **Visualization is opt-in.** The default pipeline delivers verified & analyzed
> solutions; the visualization agent runs only when the user asks to see results
> (or when "looking" helps debug a failure). Verification never depends on it —
> code-generation does in-situ NaN/Inf checks and numerical-analysis owns
> convergence. Raw solution output is still written by code-generation as the
> result data. See `docs/DECISIONS.md` (D20).

## Failure handling

- **Compile/runtime errors** are fixed by `code-generation` in its own
  build→run→fix loop (retry budget ~5; stop on a repeated error).
- **Unfixable-in-place failures** escalate via the Results Manifest's
  `escalation` block (`suspected_cause`, `target_agent`, `summary`, `attempts`);
  the orchestrator routes them to `numerical-analysis` (plan), `pde-modeling`
  (model), or resolves environment issues itself. A run that completes but
  converges wrongly is not an escalation — it flows to `numerical-analysis` for
  assessment.

## Human-in-the-loop

The orchestrator shares the session with the human, so it pauses at decision
gates rather than only at tool-permission prompts. **Default autonomy is
`interactive`** (pause at every gate); `checkpointed` gates only the model +
production runs; `autonomous` runs through and stops only on escalation. Gates:
after the Problem Spec, after the Numerical Plan, before production/large runs,
at results + interpretation, and before higher-fidelity escalation.
`open_questions` and `environment`/`unknown` escalations are always surfaced —
never silently assumed. See `.agents/pde-sim/skills/orchestration/SKILL.md` and
`docs/DECISIONS.md` (D17).

## Reuse & self-improvement (Level 1)

The system accumulates verified knowledge across runs, guarded by the MMS gate:
- **Case index** (`components/case-index.json`) — a registry of solved, verified
  studies (problem class → plan → confirmed order) the orchestrator consults
  before planning from scratch.
- **Component library** (`components/`) — reusable building blocks that passed
  MMS, promoted from completed studies.
- **Regression suite** (`tests/run_regression.sh`) — rebuilds/reruns components
  and re-validates artifacts; a promotion or skill edit is trusted only if this
  still passes. Every change is a revertible git commit, human-gated (D17).

See `docs/DECISIONS.md` (D18). Levels 2–3 (a curator step proposing skill diffs;
eval-driven plan optimization) are deferred.

## Key design decisions (open for team review)

> Full rationale, alternatives, supersessions, and open questions:
> [`docs/DECISIONS.md`](docs/DECISIONS.md).

1. **Compile/run is a tool of `code-generation`** (it holds `Bash`), not a
   separate agent — keeps the edit→compile→fix loop inside one context.
2. **Who runs code?** Orchestrator *decides/dispatches*; code-generation
   *executes* and returns the manifest.
3. **Results go to the orchestrator**, which fans them out — not Code→Vis
   directly.
4. **Visualization reports facts, not interpretation.** Numerical root-cause is
   `numerical-analysis`; model issues are `pde-modeling`.
5. **Code splits by artifact, not by "who codes".** `code-generation` owns code
   inside/linked to the simulation binary (solver, coefficient routines,
   solution output, in-situ rendering). The `visualization` agent owns
   *standalone* post-processors that read the output files (pvpython/matplotlib/
   pyvista) and runs them itself. The shared `pde-visualization` skill is the
   VTK/ParaView knowledge base both use; Vis Spec items are tagged `in_situ` or
   `post_hoc` to route them.
6. **Viz stack = VTK / ParaView + matplotlib** (working assumption — change if
   the target libraries differ).

## Running under Claude Code

The pipeline is **opt-in**: none of its skills or agents load until you invoke the
command. Start Claude Code at the repo root and run:

```
/pde-sim simulate 2-D steady heat conduction on the unit square with
homogeneous Dirichlet boundaries
```

`/pde-sim` (committed at `.claude/commands/pde-sim.md`) loads the orchestration
brief (`.agents/pde-sim/skills/orchestration/SKILL.md`) into the main session,
which then reads the specialist role prompts (`.agents/pde-sim/agents/<role>.md`) and domain
skills (`.agents/pde-sim/skills/<name>/SKILL.md`) **by path** and dispatches
each specialist as a `general-purpose` subagent. Contract artifacts are written
under a scratch `artifacts/<study-id>/` directory (add it to your local
`.git/info/exclude`).

> **Why by path, not auto-discovery.** PETSc commits a `.claude/skills ->
> ../.agents/skills` symlink, so *anything under `.agents/skills/` auto-loads into
> every Claude Code session* (this is how PETSc's own `codegraph` and `review-*`
> skills ship). The pipeline skills therefore live **outside** that directory, under
> `.agents/pde-sim/skills/`, and are loaded explicitly by the command. The four
> specialists live under `.agents/pde-sim/agents/` and are dispatched as `general-purpose`
> subagents rather than placed under `.agents/agents/` (the auto-registered path behind the
> committed `.claude/agents` symlink), so nothing about the pipeline is auto-registered either.
> Net: the only always-on footprint is the one `/pde-sim` entry in the command menu.
> (`.gitignore` lists `.claude/*`, but the committed `.claude/skills` and `.claude/agents`
> symlinks and `.claude/commands/pde-sim.md` are force-added and tracked regardless; the
> `.claude/agents` symlink points at `.agents/agents/`, which holds no pipeline specialist.)

## Repository layout

```
.agents/                        # tracked, tool-agnostic core
├── skills/                     # AUTO-LOADED Claude Code skills (via committed .claude/skills)
│   └── codegraph/ petsc-build/ petsc-configure/ petsc-docs/ petsc-lint/
│       petsc-search-docs/ petsc-test/ review-branch/ review-mr/ review-mr-post/  # PETSc's own skills (all auto-load)
└── pde-sim/                    # framework-neutral core (this directory)
    ├── agents/      # specialist role prompts (loaded by path, not registered)
    │   ├── pde-modeling.md      numerical-analysis.md
    │   └── code-generation.md   visualization.md
    ├── skills/       # pipeline domain knowledge — loaded by path, NOT auto-discovered
    │   ├── orchestration/                          #   pipeline driver (main session)
    │   ├── numerical-methods/  petsc-solvers/       #   (petsc-solvers shared)
    │   ├── pde-formulation/    pde-visualization/   #   (pde-visualization shared)
    │   └── petsc-codegen/
    ├── contracts/    # JSON Schemas for inter-agent handoffs
    ├── components/   # verified, reusable building blocks + case-index.json (reuse registry)
    ├── examples/     # complete worked studies (examples/<study-id>)
    ├── tests/        # regression suite (guards self-improvement) + schema validator
    ├── docs/         # DECISIONS.md
    └── README.md     # this file

.claude/          # Claude Code binding (committed): skills + agents symlinks + commands/pde-sim.md
artifacts/        # per-study runtime output (scratch, regenerated per run)
```

## Portability

The design is framework-neutral; only the **binding** is tool-specific. The neutral
core is `.agents/pde-sim/contracts/`, the agent role prompts in
`.agents/pde-sim/agents/`, the pipeline knowledge in `.agents/pde-sim/skills/`, and the
`examples/`, `components/`, and `tests/` under `.agents/pde-sim/` — no hardcoded
model, and prose free of tool assumptions. Everything is loaded by **explicit file
path**, so the core does not depend on any tool's auto-discovery.

The **Claude Code binding** is the committed `/pde-sim` command
(`.claude/commands/pde-sim.md`) described in
[Running under Claude Code](#running-under-claude-code): it loads the orchestration
brief, which dispatches specialists as `general-purpose` subagents. (The separate
`.claude/skills` symlink is a pre-existing PETSc convenience for its own
`codegraph`/`review-*` skills and is not part of the pipeline.)

To add another tool (e.g. **Codex** or **opencode**) as a second binding, add a thin
per-tool trigger that points at the same brief — e.g. a Codex custom prompt
(`~/.codex/prompts/pde-sim.md`) whose body is "read
`.agents/pde-sim/skills/orchestration/SKILL.md` and act as orchestrator." No
second binding is built yet (see `docs/DECISIONS.md` D15/D19).
