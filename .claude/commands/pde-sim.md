---
description: Run the PETSc PDE simulation pipeline (opt-in multi-agent orchestrator)
argument-hint: <phenomenon to simulate> [--autonomy interactive|checkpointed|autonomous]
---

You are the orchestrator for the PETSc PDE simulation pipeline. Read and follow
`.agents/pde-sim/skills/orchestration/SKILL.md`, then drive the pipeline for the
request below. Load specialist role prompts from `.agents/pde-sim/agents/<role>.md` and the
domain skills from `.agents/pde-sim/skills/<name>/SKILL.md` by path, and dispatch
each specialist as a general-purpose subagent as the brief describes.

Request: $ARGUMENTS
