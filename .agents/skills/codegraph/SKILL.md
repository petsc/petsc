---
name: codegraph
description: >-
  Use CodeGraph to navigate PETSc and avoid duplicating existing functionality. Use whenever
  writing or reviewing PETSc C, C++, or Python, especially before adding or changing a public
  API, package utility, object operation, implementation, backend, registration, or composed
  method; when finding callers and the blast radius of a change; or when comparing sibling
  implementations. Target PETSc's own index explicitly so queries use the intended graph.
  CodeGraph is optional: if the index is absent, no CodeGraph tool or CLI is available, or
  runtime dispatch is unresolved, continue with normal repository inspection without blocking.
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

- Treat source returned by `codegraph_explore` as already read; do not fetch the same source again.
  Before editing a file, read it directly — declaration blocks, `PetscFunctionBegin` pairing, and
  the `/*TEST*/` block lie outside a symbol-scoped snippet.
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
