# petsc4py

- Validate user arguments in public Python-callable methods with explicit exceptions, not `assert`, which optimized Python can remove. Internal helpers and callback trampolines may use `assert`.
- Do not add Python-side state or validation to compensate for invariants the C API cannot express or validate. Validate only what is needed to marshal Python-owned data safely, then call PETSc.
- Do not make mpi4py a required build or runtime dependency. In tests, guard its import with `try`/`except ImportError` and skip only tests that need it; the remaining suite must load and run.
- Before you add a petsc4py changelog entry, read the changelog rules in [Documentation](documentation.md). For a change that affects only petsc4py, add the entry to the `Development` section of `src/binding/petsc4py/CHANGES.rst`. Do not add it to `doc/changes/dev.md`.
