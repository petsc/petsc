# Design decisions log

A curated record of the architectural choices behind this multi-agent system —
the *why*, the alternatives rejected, and the questions still open for review.
Faster to re-read than the full design conversation. Newest supersessions noted
inline.

**Status legend:** Accepted · Superseded · Open (needs a call)

---

## D1 — Hybrid design: subagents for roles, skills for knowledge
**Status:** Accepted
**Decision:** Each specialist is a *subagent* (`.agents/pde-sim/agents/*.md`); the reusable
domain expertise lives in *skills* (`.agents/pde-sim/skills/*/SKILL.md`) the subagents load.
**Why:** Keeps *roles* (behavior, tools, handoffs) separate from *knowledge*
(testable, reusable, shareable) so each can evolve independently — the right shape
for research-quality work.
**Rejected:** (a) *pure skills* — no context isolation or parallelism, orchestration
only heuristic; (b) *pure subagents* — knowledge trapped in one role, not reusable.

## D2 — Skills named by knowledge domain, not by owning agent
**Status:** Accepted
**Decision:** `petsc-solvers`, `pde-visualization`, etc. — not `numerical-analysis-skill`.
**Why:** Two skills are genuinely *shared* (`petsc-solvers` by numerical-analysis +
code-generation; `pde-visualization` by visualization + code-generation). Naming a
shared skill after one agent would mislead.
**Rejected:** strict 1 agent : 1 skill with matching names — simpler mental model,
but loses reuse. (User chose domain-named.)

## D3 — The orchestrator is the main session, not a subagent
**Status:** Accepted
**Decision:** Orchestrator = the top-level (main) agent session, driven by the
`orchestration` skill; it dispatches the four specialists.
**Why:** Only the top-level session can dispatch subagents (subagents cannot spawn
subagents). All coordination therefore routes through the orchestrator; specialists
never call each other (clean audit trail, no nested delegation).

## D4 — JSON-Schema contracts for every handoff, passed by reference
**Status:** Accepted
**Decision:** Six `contracts/*.schema.json` define inter-agent payloads.
Agents exchange **file paths**; large data is never inlined. Artifacts live under
`artifacts/<study-id>/`.
**Why:** Machine-checkable handoffs are the robustness backbone and make the design
model-agnostic (the seam between agents is a file, not a framework). Validated by
`tests/validate.py`.

## D5 — pde-modeling outputs JSON, not prose
**Status:** Accepted (user requirement)
**Decision:** The Problem Spec is emitted as `problem-spec.json`, never a text
description.
**Why:** Its consumer (numerical-analysis) parses it; a structured contract removes
ambiguity. Generalized to all handoffs (D4).

## D6 — Compile/run is a tool of code-generation, not a separate agent
**Status:** Accepted
**Decision:** code-generation holds `Bash` and compiles/runs itself.
**Why:** The edit→compile→fix loop must stay in one context to be fast; a separate
"compile-run agent" would force a round-trip per iteration.
**Note:** The user's drafts mentioned a "compile-run agent"; folded in as a tool.

## D7 — Orchestrator decides which runs happen; code-generation executes them
**Status:** Accepted
**Decision:** code-generation runs MMS/verification autonomously; production/large
runs happen only on the orchestrator's dispatch.
**Why:** Reconciles the two drafts ("launching is the orchestrator's role" = it
*initiates*; "as a run agent, my outputs are results" = it *executes*).

## D8 — Results go to the orchestrator, which fans them out
**Status:** Accepted
**Decision:** code-generation returns a Results Manifest to the orchestrator, which
dispatches it to numerical-analysis (interpret) and visualization (render/diagnose).
**Why:** One hub, no nested delegation, clean audit trail.
**Rejected:** Code → Vis directly (David's Q3).

## D9 — Visualization reports facts; interpretation lives elsewhere
**Status:** Accepted
**Decision:** visualization emits *factual* diagnostics (NaN/Inf, ranges, plots).
Numerical root-cause is numerical-analysis; model issues are pde-modeling.
**Why:** Avoids duplicating numerical-analysis's convergence expertise; matches the
stated purpose ("*enable* a human/agent to decide"). (David's Q1.)

## D10 — Code split by artifact type (supersedes "all code by code-generation")
**Status:** Accepted — **supersedes** the original README decision #5
**Decision:** code-generation owns code *inside/linked to the simulation binary*
(solver, coefficient routines, solution output, in-situ rendering). visualization
owns *standalone post-processors* that read output files (pvpython/matplotlib/
pyvista) and **runs them itself**. Vis Spec items are tagged `in_situ` / `post_hoc`.
**Why:** Keeps VTK expertise where the VTK code is written, and the exploratory
tweak loop where the *looking* happens. The split axis is "compiled into the sim vs
reads its output files," not "who codes." (David's Q2.)
**Cost accepted:** two execution-capable agents (both hold `Bash`); reopens the
run-boundary, now spelled out in the orchestration skill. See **OQ-2**.

## D11 — Agents reference skills by explicit path (superseded in part by D19)
**Status:** Accepted — path loading is now the **primary** mechanism (D19)
**Decision:** Keep the named "Knowledge to load" directive plus the explicit
`.agents/pde-sim/skills/<name>/SKILL.md` path in each role prompt.
**Why:** Originally three redundant layers — native progressive disclosure (Skill
tool), deterministic selection (named directive), and a path fallback. Since D19 the
pipeline skills are **not** auto-discovered, so the explicit path is no longer a
fallback but the load path itself (works via plain `Read`, portable across tools).
The Skill-tool layer no longer applies to the pipeline skills. See **D19**, **OQ-3**.

## D12 — Structured failure escalation + retry budget
**Status:** Accepted
**Decision:** code-generation fixes compile/runtime errors in a bounded loop (~5
attempts, stop on a repeated error). When it can't, it fills the Results Manifest
`escalation` block (`suspected_cause`, `target_agent`, `summary`, `attempts`); the
orchestrator routes on it first. A run that *completes but converges wrongly* is not
an escalation — it flows to numerical-analysis for assessment.
**Why:** Makes hand-backs machine-routable (symmetric with
`numerical-assessment.recommended_changes`) and stops unbounded retry loops.

## D13 — Assume PETSc environment; code-generation emits makefiles
**Status:** Accepted
**Decision:** Assume `PETSC_DIR`/`PETSC_ARCH` are set and `mpiexec` is available when
MPI is needed. code-generation produces a makefile (PETSc conf variables,
`${PETSC_LIB}`, build + `run` targets) and builds/runs through it; it never
configures PETSc or hardcodes paths.
**Why:** Matches the target environment; the pattern is a standard PETSc makefile
idiom (conf variables + `${PETSC_LIB}`, build/run targets).

## D14 — The dispatch/return envelope stays convention (for now)
**Status:** Accepted (deferred)
**Decision:** The *payload* is structured (D4); the *envelope* — the orchestrator's
dispatch prompt and an agent's final text ("write the file, report its path") —
remains a prose convention, not a schema.
**Why:** Control-flow signals already live inside the contract files (status,
escalation, verdict), so the orchestrator routes by reading them. A formal
`handoff-receipt` is only worth it once dispatch is automated. See **OQ-4**.

## D15 — Portability: framework-neutral core + a per-tool binding
**Status:** Accepted (neutralized in place) · Claude Code binding only for now
**Decision:** The system is split into a framework-neutral **core** and a thin
per-tool **binding**. The core — the JSON-Schema `contracts/`, the agent role
prompts, the `skills/` domain knowledge, `examples/`, `components/`, `tests/` — is
model- and tool-agnostic: contracts now live at top-level `contracts/` (not under
`.claude/`), agent frontmatter no longer hardcodes a model, and prose avoids
Claude-isms (the Skill/Agent-tool mechanics are flagged as binding-specific, with a
file-read fallback). The **Claude Code binding** is the `.claude/` layout (agent +
skill discovery, frontmatter dialect, `settings.local.json`) and is the only binding
built so far.
**Why:** Experiments compare multiple models/tools (e.g. Codex, opencode). Keeping the
core neutral means a second binding is additive.
**Update (D19):** the Claude Code binding is now a single committed command
(`.claude/commands/pde-sim.md`), not agent/skill auto-discovery. All pipeline content
loads by explicit path, so a second binding is just a thin trigger pointing at the same
orchestration brief (e.g. a Codex custom prompt at `~/.codex/prompts/pde-sim.md`). The
pre-existing `.claude/skills` symlink is a PETSc convenience for its own
`codegraph`/`review-*` skills only — unrelated to the pipeline.
**Deferred:** a second (Codex/opencode) binding and a source→binding generator. See
**OQ-5**.

## D16 — Visualization stack = VTK / ParaView (+ matplotlib)
**Status:** Open (assumption)
**Decision:** Assume VTK/ParaView for field rendering, matplotlib for line/
convergence plots (the `XXX` placeholders resolved from David's "VTK knowledge
bases" remark).
**Why:** Best inference available; easy to change. See **OQ-6**.

## D17 — Human-in-the-loop: approval gates, default interactive
**Status:** Accepted
**Decision:** The orchestration skill defines explicit HITL gates — after the
Problem Spec, after the Numerical Plan, before production/large runs, at
results/interpretation, and before higher-fidelity escalation — plus three
autonomy levels: **`interactive` (default)**, `checkpointed`, `autonomous`. The
Problem Spec's `open_questions`, unresolved ambiguities, and `environment`/
`unknown` escalations are surfaced to the human at *every* level. Autonomy is a
runtime policy set in the invocation, not a contract field.
**Why:** Research-quality work needs a human to sign off on the model and
authorize expensive runs — decision-level gates, not just Claude Code's
tool-permission prompts. The orchestrator = main session (D3) makes
pausing-to-ask free.
**Rejected:** a dedicated `study-config` contract field for autonomy — kept it a
runtime policy to avoid schema surface (consistent with D14).

## D18 — Level-1 self-improvement: case index + component library + regression suite
**Status:** Accepted
**Decision:** Accumulate verified knowledge across runs, guarded by the MMS gate:
a **case index** (`components/case-index.json`, schema in `contracts/`) of solved,
verified studies the orchestrator consults before planning; a **component
library** (`components/`) of MMS-passed building blocks code-generation reuses;
and a **regression suite** (`tests/run_regression.sh`) that rebuilds/reruns
components and re-validates artifacts. Promotions and skill edits are human-gated
(D17) and trusted only if the suite still passes; every change is a revertible
git commit. Levels 2–3 (a curator step proposing skill diffs; eval-driven plan
optimization) are deferred.
**Why:** This domain has verifiable ground truth (MMS/convergence), so reuse can
be *validated* rather than hoarded — safe, auditable improvement instead of
prompt drift.
**Rejected:** unsupervised self-editing of skills/prompts — drift and error
accumulation with no arbiter.

## D19 — Opt-in: pipeline skills off the auto-discovery path; command is the trigger
**Status:** Accepted — supersedes the auto-load behavior implied by D1/D3/D11
**Decision:** The six pipeline skills live under `.agents/pde-sim/skills/`, **not**
under `.agents/skills/`. PETSc commits a `.claude/skills -> ../.agents/skills` symlink,
so anything under `.agents/skills/` auto-loads into *every* Claude Code session; the
pipeline skills are kept out of it. The Claude Code binding is a single committed
command, `.claude/commands/pde-sim.md` (`/pde-sim`), which loads the orchestration brief;
the orchestrator then reads role prompts (`.agents/pde-sim/agents/<role>.md`) and skills by path
and dispatches specialists as `general-purpose` subagents. The pipeline's role prompts live under
`.agents/pde-sim/agents/`, off the auto-registration path: the committed
`.claude/agents -> ../.agents/agents` symlink exposes only `.agents/agents/` (currently just
`petsc-search-docs`), so no pde-sim specialist is auto-registered.
**Why:** The pipeline should engage only when a user asks for it. The prior placement
would have auto-loaded six skills' metadata (~3.5k tokens) and made them
model-invokable in every repo session — contradicting the opt-in intent and the README.
The only always-on footprint is now the one `/pde-sim` menu entry.
**Cost accepted:** `general-purpose` workers get the full toolset (no per-role `tools:`
scoping). Alternative (rejected for now): place the specialists under `.agents/agents/`
for native subagents, which reintroduces always-on registration.
**Note:** PETSc's own skills under `.agents/skills/` — `codegraph`, `petsc-build`,
`petsc-configure`, `petsc-docs`, `petsc-lint`, `petsc-search-docs`, `petsc-test`,
and `review-branch`/`review-mr`/`review-mr-post` — auto-load as before; intended
and unrelated to the pipeline.

## D20 — Visualization is opt-in / on-demand
**Status:** Accepted — refines D8
**Decision:** The visualization agent is not a mandatory stage. The default
pipeline delivers **verified & analyzed** solutions; the orchestrator dispatches
visualization only when the user asks to *see* results (plots, renders, fields),
or when it needs "looking" to debug a failure. When visualization is not in play,
the orchestrator skips the pre-run `vis-spec.json` and the post-run
`analysis-report.json` fan-out. Raw solution output is still written by
code-generation as the result data — that is unaffected.
**Why:** Rendering is a user-facing deliverable, wanted only on request.
Verification never depends on it: code-generation does in-situ NaN/Inf checks and
numerical-analysis owns convergence interpretation from the manifest's
`convergence_study`, so the visualization agent's factual diagnostics are
redundant for correctness. Previously D8 and the README treated "visualized" as
part of the definition of done and Stage 4 fanned the manifest to visualization
unconditionally — an inconsistency (pre-run vis-spec was already "optional").
**Refines (does not reject) D8:** results still flow through the orchestrator,
which fans them out — but the visualization arm of that fan-out is conditional.
**Cost accepted:** the orchestrator must judge "did the user ask to see results?"
rather than always rendering; ambiguous cases are resolved at HITL gate 4.

---

## Open questions (for David / the team)

- **OQ-1 — David's three questions:** interpretation split (D9), viz code location
  (D10), and results routing (D8) are resolved as above. **Confirm or override.**
- **OQ-2 — Execution surface area.** D10 intended only code-generation and
  visualization to hold `Bash`, but **D19 supersedes that premise**: specialists
  dispatch as `general-purpose` subagents with the full toolset, so *all four* are
  execution-capable and the per-role `tools:` frontmatter is advisory, not enforced.
  Open question stands: keep the wide surface area, or reintroduce per-role scoping
  (which under D19 means committing native `.agents/agents/` subagents)?
- **OQ-3 — Do subagents receive project skills via the Skill tool?** **Resolved/moot
  (D19):** the pipeline no longer uses native subagents or skill auto-discovery —
  specialists are `general-purpose` subagents and all skills load by explicit path.
- **OQ-4 — Formalize the envelope?** Add a `handoff-receipt` contract when we start
  automating dispatch (e.g. via the Workflow tool's structured-output option)?
- **OQ-5 — Cross-model support.** If/when we want Codex/GPT/Gemini: add `AGENTS.md`
  + a model-agnostic runner. Priority?
- **OQ-6 — Confirm the viz libraries.** Is VTK/ParaView correct, and what exactly
  were the `XXX` packages?
