# PETSc

PETSc is a C library for parallel numerical computation using MPI. The codebase is primarily C, with Python bindings in `src/binding/petsc4py/`.

This file must be self-contained. Do not rely on linked Markdown files being read automatically. The essential repo guidance is embedded below.

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

## Core Working Rules

- Preserve PETSc style and naming conventions.
- Keep edits minimal and local to the requested change.
- Match existing patterns in the package you are modifying before introducing a new one.
- Avoid unnecessary code duplication. Prefer reusing or extending nearby logic when it keeps behavior clear and local.
- Do not add speculative abstractions or broad refactors unless explicitly requested.
- If you modify source, check whether test blocks, expected output files, or documentation need corresponding updates.
- Never call `PetscFinalize()` inside an `if (...)` block, including early-return patterns like `if (flag) { ...; PetscFinalize(); return 0; }`. Arrange control flow so finalization happens exactly once on every normal exit path.

## PETSc Naming And API Conventions

- Public function names use capitalized components, for example `KSPSolve()` and `MatGetOrdering()`.
- Enum constants and macros are uppercase with underscores, for example `MAT_FINAL_ASSEMBLY`.
- Private PETSc functions generally end in `_Private` or include an implementation suffix such as `MatMult_SeqAIJ`.
- Implementation functions should start with the interface operation name and then the implementation name, for example `KSPSolve_GMRES()`.
- Options database keys are lowercase with underscores and usually mirror the setter name without `set`, for example `-ksp_gmres_restart`.
- Function typedef names should end in `Fn`.

## PETSc Data Type Rules

- Use `PetscInt` for most indices and array lengths.
- Use `PetscCount` for sizes or counts that may exceed `PetscInt`.
- Use `size_t` for memory sizes in bytes, not logical array lengths.
- Do not silence narrowing warnings with blind casts. Use PETSc cast helpers such as `PetscIntCast()` when converting to narrower integer types.
- Prefer PETSc MPI wrappers that accept PETSc count types when large counts may be involved.

## C Coding Style

- Formatting is controlled by `.clang-format`. Use `make clangformat` when needed.
- CI also checks source rules with `make checkbadSource`.
- Header prototypes should not include parameter names, but function typedef declarations should.
- Group local variables by type. Do not mix pointer arities on the same declaration line.
- Initialize local variables in the declaration when practical.
- In PETSc routines, place exactly one blank line between local declarations and `PetscFunctionBegin`.
- PETSc example functions, including `main()`, should begin with `PetscFunctionBeginUser` after declarations.
- Functions that begin with `PetscFunctionBegin` must return with `PetscFunctionReturn(...)` or `PetscFunctionReturnVoid()`, not raw `return`.
- For `PetscErrorCode` functions, return `PetscFunctionReturn(PETSC_SUCCESS)` on success.
- Wrap PETSc calls with `PetscCall(...)`. For external library calls, use the appropriate PETSc wrapper such as `PetscCallExternal()` or package-specific variants.
- Single-statement `if`/`else` blocks must omit braces.
- Do not leave commented-out code or dead `#ifdef` blocks in source files.
- Use `/* ... */` for multiline comments and `// ...` for short single-line comments.
- Do not decorate multiline comments with leading `*` on each line.
- Use correct grammar and spelling in comments and messages.
- Follow C90-style declarations at the start of a block, except loop indices declared in `for (...)` initializers and small-scope declarations inside nested blocks when appropriate.

## Error Handling And PETSc Idioms

- Most PETSc functions return `PetscErrorCode`.
- Use `PetscFunctionBegin`/`PetscFunctionBeginUser` and `PetscFunctionReturn(...)` consistently.
- Check object validity and arguments using the usual PETSc validation macros when working in code paths that already use them.
- Reuse existing PETSc utility routines and macros before adding custom helpers.
- Do not wrap `PetscCheck()` in an outer `if (...)` when the condition can be expressed directly in the check. Prefer a single guard such as `PetscCheck(!use_mms || sw->Ax == sw->Ay, ...)` over `if (use_mms) PetscCheck(sw->Ax == sw->Ay, ...)`.

## Testing Requirements

- PETSc tests are described in `/*TEST ... TEST*/` blocks at the bottom of source files.
- If behavior changes, update the test block and expected output files under the local `output/` directory when needed.
- Common test block keys include:
  - `test` or `testset`
  - `suffix`
  - `nsize`
  - `args`
  - `requires`
  - `output_file`
  - `filter` and `filter_output`
  - `localrunfiles`
  - `temporaries`
  - `timeoutfactor`
  - `env`
- Use `requires:` for runtime requirements such as packages, precision, `!complex`, or `datafilespath`.
- Expected output normally lives in `output/<testname>.out` relative to the source file.
- Keep tests targeted. Add or update the narrowest test that proves the behavior you changed.

## Build And Test Commands

- `make clangformat` - format source
- `make checkclangformat` - verify formatting
- `make checkbadSource` - run PETSc source-style checks
- `make test search='<pattern>'` - run tests matching a pattern
- `make alltests TIMEOUT=600` - run the full suite with an extended timeout
- `make branch-review CLAUDE_OPTS='<options>'` - run AI-assisted review on the current branch

## Merge Request Expectations

- All changes are expected to arrive through GitLab merge requests.
- Keep diffs reviewable and focused.
- Before concluding work, consider whether formatting, source-style checks, and at least one relevant test should be run.
- If you cannot run the appropriate verification in the current environment, say so explicitly.

## Practical Agent Guidance

- Read nearby code before editing so new code matches local conventions.
- When touching PETSc C code, check for consistent use of `PetscCall`, `PetscFunctionBegin`, naming, and test coverage.
- When touching tutorials or tests, inspect neighboring files for the expected `/*TEST*/` structure and output-file conventions.
- When touching public interfaces, check whether headers, docs, and examples need updates.
- Prefer citing exact file paths and commands in your responses.

## Anti-Patterns (MUST avoid when writing or reviewing)

### PetscFinalize inside conditional

WRONG — finalization inside an early-return `if`:
```c
if (test_spatial_order) {
  PetscCall(TestSpatialOrder(comm, &sw));
  PetscCall(PetscFinalize());
  return 0;
}
```
RIGHT — finalization on the single exit path:
```c
if (test_spatial_order) PetscCall(TestSpatialOrder(comm, &sw));
else PetscCall(RunForwardModel(comm, &sw));
PetscCall(PetscFinalize());
return 0;
```

### Braces on single-statement if/else

WRONG:
```c
if (radius <= 0.0) {
  return 0.0;
}
```
RIGHT:
```c
if (radius <= 0.0) return 0.0;
```

This also applies to `else` blocks paired with multi-statement `if` — check each branch independently:
WRONG:
```c
  if (type == TYPE_A) {
    stmt1;
    stmt2;
  } else {
    SETERRQ(comm, PETSC_ERR_SUP, "unsupported");
  }
```
RIGHT:
```c
  if (type == TYPE_A) {
    stmt1;
    stmt2;
  } else SETERRQ(comm, PETSC_ERR_SUP, "unsupported");
```

## Key References

External links are for human convenience only; do not assume linked Markdown files will be ingested automatically.

- Development docs: https://petsc.org/main/overview/
- Release docs: https://petsc.org/release/overview/
- GitLab project: https://gitlab.com/petsc/petsc
