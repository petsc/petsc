# petsc4py

- Validate user arguments in public Python-callable methods with explicit exceptions, not `assert`, which optimized Python can remove. Internal helpers and callback trampolines may use `assert`.
- Do not add Python-side state or validation to compensate for invariants the C API cannot express or validate. Validate only what is needed to marshal Python-owned data safely, then call PETSc.
- Do not make mpi4py a required build or runtime dependency. In tests, guard its import with `try`/`except ImportError` and skip only tests that need it; the remaining suite must load and run.
