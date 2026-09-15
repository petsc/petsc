---
name: codegraph
description: >-
  Use CodeGraph when `.codegraph/` exists to navigate PETSc and avoid duplicating existing
  functionality. Use whenever writing or reviewing PETSc C, C++, or Python, especially before
  adding or changing a public API, package utility, object operation, private object state or
  lifecycle, implementation, backend, registration, or composed method; when finding callers
  and the blast radius of a change; or when comparing sibling implementations. Target PETSc's
  own index explicitly so queries use the intended graph. If `.codegraph/` is unavailable,
  continue with normal repository inspection.
---

# CodeGraph for PETSc

Use CodeGraph to follow PETSc's interface/implementation structure and honor the repository rule
to reuse or extend existing routines before adding new ones. Prefer `codegraph_explore`; it
returns relevant source, relationships, callers, and blast-radius information in one call.

## Select the PETSc index

Use whichever CodeGraph interface is available, and always target the PETSc repository root:

- With the `codegraph_explore` tool, pass the PETSc root as `projectPath`.
- With shell access, confirm the index with `codegraph status <petsc-root>`, then run
  `codegraph explore "..."` from the PETSc root so the query resolves against PETSc's index.
- Confirm `<petsc-root>/.codegraph/` exists before querying it.

Do not silently use an enclosing repository's index. If the PETSc index is absent, or no
CodeGraph tool or CLI is available, skip CodeGraph, continue with normal repository inspection,
and (if the index is what's missing) mention that the user can run `codegraph init` from the
PETSc root. Indexing is the user's decision; do not initialize it automatically.

## Search before implementing

Before adding a function, helper, object operation, or implementation:

1. Name the proposed symbol and state its behavior.
2. Query by both likely names and the behavior it provides. Include the PETSc package and data
   type to disambiguate common names.
3. Compare the closest existing implementation and at least one sibling in the same package or
   backend family.
4. Reuse an equivalent routine. Prefer a small extension or fix when an existing routine is
   nearly capable. Add a new routine only when semantics, performance, or compatibility genuinely
   differ.

Common reinvention hot spots include allocation and array helpers, integer casts, sorting and hash
utilities, string and options handling, object composition, viewers, logging, and package-private
`*_Private` routines.

## Trace PETSc runtime dispatch

PETSc frequently dispatches through macros, operation tables, composed functions, and registries.
CodeGraph may find both endpoints without connecting the runtime edge. Do not interpret a missing
edge as proof that the symbols are unrelated.

For `PetscUseTypeMethod()` and `PetscTryTypeMethod()`:

1. Query the public interface routine, operation-field name, likely implementation names, and
   constructor or operation-table symbol together. For example, query `MatMult mult
   MatMult_SeqAIJ MatCreate_SeqAIJ MatOps_Values`.
2. Follow the interface call to `obj->ops->operation`.
3. Locate assignments or table entries for that operation and inspect the implementation
   functions they select.
4. For repeated file-local table names such as `MatOps_Values`, include the implementation path
   in the query and verify the table in that file. Map positional initializers against the
   corresponding `struct _*Ops` field order.
5. Continue through subtype, backend, and options-driven setup that may override the base
   operation table.
6. Independently resolve the assignment or table step when CodeGraph stops at the function
   pointer.

For `PetscUseMethod()` and `PetscTryMethod()`:

1. Query the method key, the calling interface, `PetscObjectComposeFunction()`, and likely
   implementation names together.
2. Match the exact composed-function key, including its `_C` suffix.
3. Check every relevant type constructor and conversion path that composes or removes the method.

For registered types, query the public creation path, `*Register()` routine,
`PetscFunctionListAdd()`, type name, and implementation constructor together. Expect the
registry-selected function-pointer call to require verification through the exact type key and
registration entry.

## Trace object lifecycle and cached state

When changing private object data, defaults, cached state, or setup behavior, query the private
implementation header together with the exact class-prefixed lifecycle routines: `*Create()`,
`*SetType()`, `*SetFromOptions()`, `*SetUp()`, `*Reset()`, `*Destroy()`, and `*View()`. Include
`*Load()`, `*Duplicate()`, `*Copy()`, or `*Convert()` when the object provides them.

Use exact symbols instead of bare lifecycle words such as `setup`, `reset`, or `destroy`, which
match unrelated packages and languages. For example, query `PCCreate PCSetType PCSetFromOptions
PCSetUp PCReset PCDestroy PCView pcimpl.h` as one bundle.

Verify each applicable part of the lifecycle:

- Initialize defaults and ownership in the base creation path and implementation constructor.
- Reinitialize or transfer state correctly when `*SetType()` destroys one implementation and
  installs another.
- Invalidate setup flags, object states, and cached results from every setter or dependency change
  that affects them.
- Release owned resources in reset, destroy, and type-change paths without breaking reference
  counting.
- Preserve or expose the state consistently through options, view, load, duplicate, copy, and
  conversion paths where those operations exist.

CodeGraph may not represent C struct-member references as complete field edges. After exploring
the object and its lifecycle, directly search the exact `->member` and `.member` tokens for every
read and write of a changed member. This closes a known graph-coverage gap; it is not a reason to
re-verify source CodeGraph already returned.

## Assess blast radius

Before changing a shared symbol, use CodeGraph to find callers, references, siblings, and covering
tests. Supplement the result where PETSc's runtime structure can hide dependencies:

- Search operation-table assignments and composed-function keys for dispatch changes.
- Search registration lists and type constructors for implementation changes.
- Check public headers, documentation, bindings, examples, and tests for public API changes;
  generated or conditionally compiled surfaces may not appear as static graph edges.
- Check CPU, MPI, CUDA, HIP, and Kokkos siblings when changing behavior shared across backends.

Use these checks to scope the requested change, not to expand it into unrelated cleanup.

## Practical rules

- For documentation-only verification of a known symbol, option, prototype, or source path, use
  direct source search. Use CodeGraph when dispatch, lifecycle, callers, or implementation
  relationships must be understood.
- Treat source returned by `codegraph_explore` as already read for understanding; do not reopen the
  same symbol merely to verify it. Before editing, read the containing file for surrounding context
  omitted from symbol-scoped snippets, such as declarations, `PetscFunctionBegin` pairing, and the
  `/*TEST*/` block. This read supplies editing context rather than re-validating CodeGraph's source.
- Review evidence must come from the reviewed revision, which may differ from the graph or
  working tree. Follow [review verification](../review-mr/review-procedure.md#4-verify-each-finding-before-reporting).
- Use other repository-inspection capabilities only for details CodeGraph did not cover,
  especially macro expansion, function-pointer assignment, generated files, preprocessor
  variants, and text-only configuration.
- Keep queries specific. Combine interface, operation, type, constructor, and implementation names
  when tracing dispatch.
- If CodeGraph reports pending re-indexing for an edited file, read that file directly until the
  index catches up. If it reports that auto-sync is disabled, directly verify all potentially
  changed files.
- Use the compiler, linter, and relevant tests for correctness. CodeGraph describes structure and
  impact; it does not validate behavior.
