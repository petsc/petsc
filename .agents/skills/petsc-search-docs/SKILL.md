---
name: petsc-search-docs
description: Answer a PETSc question from the locally built docs (manual pages, users manual, FAQ, install guide) with a citation. Use when the user asks what a function, type, or option does, how to use a feature, or where something is documented, or runs `/petsc-search-docs`. Not for editing code.
argument-hint: <question or PETSc identifier>
---

The question is `$ARGUMENTS`, or the user's last question if empty. Do not answer it
from memory or from the source tree.

Dispatch the `petsc-search-docs` subagent with the question verbatim, plus any context
from this conversation that it would otherwise lack, in one self-contained prompt.

Relay its answer and its `Sources:` list unchanged, including the `Doc build:` line. Do
not re-read the pages it cited or re-derive its excerpts.
