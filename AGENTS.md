# PETSc

PETSc is a C library for parallel numerical computation using MPI. The codebase is primarily C, with Python bindings in `src/binding/petsc4py/`.

Shared rules are below; optional skills provide task procedures through the coding tool's skill system. Respect the user's skill settings, including for references between skills; do not load disabled skills through direct file reads. Read conditional convention references explicitly when relevant.

## Project Layout

- `src/<package>/` - source organized by package: `vec`, `mat`, `ksp`, `snes`, `ts`, `dm`, `tao`, `sys`
- `src/<package>/impls/` - concrete implementations of each package's abstract type
- `src/<package>/interface/` - public API for each package
- `src/<package>/tests/` and `src/<package>/tutorials/` - tests and examples
- `include/` - public headers such as `petsc*.h`
- `include/petsc/private/` - private headers such as `*impl.h`
- `src/binding/petsc4py/` - Python bindings and packaging logic
- `config/` - configure, build, and test harness generation
- `doc/` - user and developer documentation
- `.agents/pde-sim/` - opt-in multi-agent PDE simulation pipeline; in Claude Code run `/pde-sim <phenomenon>` to start it (nothing loads until invoked)

## Core Working Rules

- Keep edits minimal and local to the request. Read nearby code, preserve PETSc style, and reuse or extend existing logic. Do not add speculative abstractions or broad refactors unless requested.
- When several APIs make the same decision, for example which implementation or strategy to select, implement it once in a private helper that also reports what it chose, and call that helper from each API.
- Use PETSc accessors for object metadata instead of reconstructing internal layouts in callers or bindings. If metadata is missing, extend the owning API and document returned data's size, ownership, and lifetime. If only the documentation is unclear, clarify it.
- Fix only what the task requires. Report other improvements you notice, in scripts, makefiles, tools, or documentation, to the user separately with the file path and expected benefit; do not put them in the patch, review findings, or MR comments unless asked.
- For source changes, check test blocks, expected outputs, and documentation; include headers and examples for public interfaces. Run relevant checks and tests.
- Final test changes must exercise successful behavior: do not add expected-error cases, assertions of expected error codes, or `PetscPushErrorHandler()`/`PetscPopErrorHandler()` calls.
- The user selects `PETSC_ARCH`; reuse the value already supplied for the task. Ask if it is missing before configuring or building PETSc or petsc4py, running PETSc-dependent tests or executables, or importing petsc4py; pass it explicitly and never infer or change it unless asked. Architecture-independent checks need no architecture. Targets managing a dedicated default architecture, such as `make docs`, are exempt when using that default.
- A documentation audit or review does not authorize a documentation build; run it only when explicitly requested or approved for the task.
- All changes are expected to arrive through focused, reviewable GitLab merge requests.

## Conditional Conventions

Before working on any of the following, read its convention reference. Load only the references
relevant to the task, including during reviews and when no build or lint check will run.
Reread this file or a convention reference only if it changed or is no longer in context.

| Task | Reference |
| --- | --- |
| New features, API changes, or documentation edits/reviews | [Documentation](.agents/conventions/documentation.md) |
| C API docstrings (`/*@ ... @*/`) | [C docstrings](.agents/conventions/c-docstrings.md) |
| Kokkos code | [Kokkos](.agents/conventions/kokkos.md) |
| petsc4py source or tests | [petsc4py](.agents/conventions/petsc4py.md) |

## Writing PETSc Contribution Materials

Apply these rules to PETSc contribution materials and their drafts: commit messages, MR
descriptions, review reports and comments, documentation, and code comments. They do not govern
unrelated conversations or prescribe the user's conversational style.

- Lead with the result and why it matters. Make the text understandable without the drafting conversation; include relevant verification, limitations, and reproduction details.
- Write clear, grammatical, concise prose for peer mathematicians and engineers. Explain unfamiliar terms, preserve PETSc terminology and exact API names, commands, and diagnostics, and remove repetition and boilerplate before presenting.
- Prefer self-explanatory code. Use comments and docstrings for non-obvious behavior, correctness constraints, rationale, or required documentation. Preserve accurate comments; omit process narration and edit history.

## PETSc Naming And API Conventions

- Public function names use capitalized components, for example `KSPSolve()` and `MatGetOrdering()`.
- Enum constants and macros are uppercase with underscores, for example `MAT_FINAL_ASSEMBLY`.
- Private PETSc functions generally end in `_Private` or include an implementation suffix such as `MatMult_SeqAIJ`.
- Implementation functions should start with the interface operation name and then the implementation name, for example `KSPSolve_GMRES()`.
- Options database keys are lowercase with underscores and usually mirror the setter name without `set`, for example `-ksp_gmres_restart`.
- Options-line syntax (applies in **all** documentation — `/*@ @*/` blocks, `/*MC M*/` blocks, `.md` chapters, and prose comments):
  - Enumerated values use `(choice1|choice2|choice3)`, **never** `<a, b>` or `{a, b}`.
  - Free-form arguments use plain words (e.g. `radius`, `size`, `name`), **never** `<radius>` and never backticks around the arg.
  - Inside the `Options Database Keys:` linter block (the bullet entries `+ -opt val - desc`), the option-name is **bare** — no backticks. Example: `. -petscda_letkf_localization_type (none|gaspari_cohn|gaussian|boxcar) - select the localization kernel`.
  - In inline prose elsewhere (Notes blocks, `.md` chapters, `/*MC M*/` body), wrap the option in backticks as code: `` `-petscda_type name` ``, `` `-log_view` ``. This matches the convention used throughout `doc/manual/` and PETSc's docstring prose.
  - Backticks are for rendered documentation only. In ordinary `/* */` and `//` comments, including the header comment of a tutorial or test, the option-name stays **bare** — no tutorial or test in the tree backticks one. The enumerated-value and plain-word rules above still apply there.
- Function typedef names should end in `Fn`.
- `MPI_Comm_size()` → local `size`; `MPI_Comm_rank()` → `rank`. No prefixed variants (`comm_size`, `nprocs`). If `size` is taken, rename the other local.
- Reserve the `_p_` prefix for struct tags associated with PETSc objects, such as `_p_Mat`; other struct tags must not use it.

## PETSc Data Type Rules

- Use `PetscInt` for most indices and array lengths.
- Use `PetscCount` for sizes or counts that may exceed `PetscInt`.
- Use `size_t` for memory sizes in bytes, not logical array lengths.
- Do not silence narrowing warnings with blind casts. Use PETSc cast helpers such as `PetscIntCast()` when converting to narrower integer types.
- Prefer PETSc MPI wrappers that accept PETSc count types when large counts may be involved.

## C Coding Style

- Format changed C/C++ files in PETSc's formatting set with the repository's `.clang-format` and required clang-format version.
- Header prototypes should not include parameter names, but function typedef declarations should.
- The declaration block at the top of a routine or nested scope is one contiguous group: variables grouped by type (all `PetscInt`s adjacent, all `PetscReal`s adjacent, etc.), no mixed pointer arities on a single line, no blank lines or section comments splitting the block. Initialize in the declaration when practical. Exactly one blank line separates the block from the first statement, including `PetscFunctionBegin`/`PetscFunctionBeginUser` at routine scope.
- In PETSc tutorials and tests, `main()` and all functions returning `PetscErrorCode` must begin with `PetscFunctionBeginUser` after declarations.
- Functions that begin with `PetscFunctionBegin` must return with `PetscFunctionReturn(...)` or `PetscFunctionReturnVoid()`, not raw `return`.
- For `PetscErrorCode` functions, return `PetscFunctionReturn(PETSC_SUCCESS)` on success.
- Wrap PETSc calls with `PetscCall(...)`. For external library calls, use the appropriate PETSc wrapper such as `PetscCallExternal()` or package-specific variants.
- Do not leave commented-out code or dead `#ifdef` blocks in source files.
- Use `/* ... */` for multiline comments and `// ...` for short single-line comments.
- Do not decorate multiline comments with leading `*` on each line.
- Always append `()` to function names when mentioning them in comments, for example `MatAssemblyEnd_MPIAIJ()`.
- Follow C90-style declarations at the start of their enclosing block. Prefer declaring variables used only within a genuinely new nested `{ ... }` scope at the beginning of that scope. The only other allowed exception is a loop index in a `for (...)` initializer. Do **not** sprinkle `const T x = ...;` lines between statements, including after an early-return guard.

### Braces on single-statement if/else

Omit braces around any `if`, `else if`, or `else` branch whose body is one statement. Check each
branch independently, including an `else` paired with a multi-statement `if`:

```c
// Incorrect
if (type == TYPE_A) {
  stmt1;
  stmt2;
} else {
  SETERRQ(comm, PETSC_ERR_SUP, "unsupported");
}
// Correct
if (type == TYPE_A) {
  stmt1;
  stmt2;
} else SETERRQ(comm, PETSC_ERR_SUP, "unsupported");
```

## Error Handling And PETSc Idioms

- Most PETSc functions return `PetscErrorCode`.
- Check object validity and arguments using the usual PETSc validation macros when working in code paths that already use them.
- Do not wrap `PetscCheck()` in an outer `if (...)` when the condition can be expressed directly in the check. Prefer a single guard such as `PetscCheck(!use_mms || sw->Ax == sw->Ay, ...)` over `if (use_mms) PetscCheck(sw->Ax == sw->Ay, ...)`.
- Do not call `MatAssemblyBegin()`/`MatAssemblyEnd()` after `MatDenseRestoreArray*()` or `MatDenseRestoreColumnVec*()`. The Get/Restore pair is the assembled write path for dense matrices — the matrix stays assembled across it. Adding "just to be safe" assembly is wrong, not defensive. Assembly is only needed after `MatSetValues()`-style entry, where deferred stashing actually requires a flush.

### PetscFinalize inside conditional

Never call `PetscFinalize()` inside an `if` block. Arrange one finalization on every normal exit
path, including when a mode bypasses the main computation:

```c
// Incorrect
if (test_spatial_order) {
  PetscCall(TestSpatialOrder(comm, &sw));
  PetscCall(PetscFinalize());
  return 0;
}
// Correct
if (test_spatial_order) PetscCall(TestSpatialOrder(comm, &sw));
else PetscCall(RunForwardModel(comm, &sw));
PetscCall(PetscFinalize());
return 0;
```

## Key References

External links are for human convenience only; do not assume linked Markdown files will be ingested automatically.

- Development docs: https://petsc.org/main/overview/
- Release docs: https://petsc.org/release/overview/
- GitLab project: https://gitlab.com/petsc/petsc
