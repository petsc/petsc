---
name: petsc-test
description: >-
  Define, discover, select, run, and debug PETSc harness tests or petsc4py Python tests in a
  user-supplied PETSC_ARCH, and update expected output. Use when validating a source change
  or reproducing a test failure; documentation checks are covered by petsc-docs.
---

# Test PETSc and petsc4py

`arch-name` is the user-supplied architecture. Run PETSc commands from the repository root and
petsc4py commands from `src/binding/petsc4py/`.

Keep test runs relevant to the change. Do not run `make test` without a selector, `make alltests`,
or the complete petsc4py suite with an unfiltered `python test/runtests.py` unless the user
requests the full suite. If tests need capabilities missing from the supplied architecture,
report them; do not change the configuration implicitly.

## Define tests

Follow neighboring test blocks and expected-output conventions when adding or updating tests.

PETSc harness tests are described in `/*TEST ... TEST*/` blocks at the bottom of source files.
Common test-block keys are `test`, `testset`, `suffix`, `nsize`, `args`, `requires`,
`output_file`, `filter`, `filter_output`, `localrunfiles`, `temporaries`, `timeoutfactor`, and `env`.
Use `requires:` for runtime requirements such as packages, precision, `!complex`, or
`datafilespath`. Expected output normally lives in `output/<testname>.out` relative to the source
file.

Reuse an existing test source when possible, adding a test variant with arguments drawn from
nearby tests or generated scripts. Create a source only if none can naturally exercise the
behavior or the user explicitly requests a new one. Harness tests apply metadata and expected-output
comparison; for focused diagnosis or an unregistered scenario, run the executable directly and
state what it validates. Use `testset` inheritance for shared setup and loops for related variants.
Use separate suffixes for distinct tests. Loops share output by default; use
`{{X Y}separate output}` for a script and expected-output file per value. Do not invent broad
option matrices merely to increase coverage.

## Select tests

Use the repository-root `make` entry point. Its makefiles forward targets to the configured make
(`OMAKE`) when needed, including when the system make is GNU Make 3.x. Do not bypass this
forwarding with `make -f gmakefile` using the system make. If an older checkout lacks the
forwarding rule, read `OMAKE` from `arch-name/lib/petsc/conf/petscvariables` and use that configured
executable with `-f gmakefile` for the test targets below.

Test definitions are generated as needed. Use `print-test` to inspect a selection before its
first run, then use the same selectors with `make test`. Reuse the inspected selection while
the selectors, test definitions, and architecture configuration are unchanged; repeat
`print-test` when any of them changes or the selection is no longer in context. Recheck dynamic
selections such as `test-fail=1` after each test run.

```console
# Select by source path, directory, or target glob
$ make PETSC_ARCH=arch-name print-test s='src/ksp/ksp/tests/ex1.c'

# Narrow that selection by target name
$ make PETSC_ARCH=arch-name print-test s='src/ksp/ksp/tests/ex1.c' i='*ex1_1'

# Find tests exercising an external package
$ make PETSC_ARCH=arch-name print-test query='requires' queryval='hdf5'

# Find all source definitions requiring GPU-aware MPI, including unavailable tests
$ ./config/query_tests.py --use-source --petsc-dir=. requires '*GPU_AWARE*'

# Query /*TEST*/ args (leading option dashes are omitted)
$ make PETSC_ARCH=arch-name print-test query='args' queryval='*pc_type*bddc*'

# Show all harness options
$ make PETSC_ARCH=arch-name help-test
```

After inspecting a selection, run it with the same selectors, for example:

```console
$ make PETSC_ARCH=arch-name test s='src/ksp/ksp/tests/ex1.c' i='*ex1_1'
```

An empty or entirely skipped selection does not validate the change. Check the selected tests
and their requirements before reporting success.

`query` names a test-level `/*TEST*/` keyword and `queryval` matches its value using globs.
Matches across distinct field values are combined. Queries inspect test definitions, not
source code or runtime types. `query='requires'` searches test-level requirements, not
`build: requires:`.

`make` queries use the test metadata generated for the active `PETSC_ARCH`.
`config/query_tests.py --use-source --petsc-dir=.` reads source test definitions without requiring
a configured architecture and can return tests the active configuration cannot build or run.
Argument queries omit leading dashes and numeric text, but retain non-numeric loop values.
Inspect test definitions for exact argument values and input files. Use a target glob such as
`s='*bddc*'` for broader name-based candidates; verify runtime use in source when needed.

Targeted controls: `NO_RM=1` retains executables; `PRINTONLY=1` prints without running (only the
first command for loops); `V=1` shows loop commands; `EXTRA_OPTIONS=` appends options; `DEBUG=1`
and `VALGRIND=1` instrument runs; and `test-fail=1` reruns previous failures. See
`doc/developers/testing.md` for generated scripts and complex queries; inspect
`config/query_tests.py` only when selector implementation details are needed.

## Diagnose failures

Use `make PETSC_ARCH=arch-name print-test test-fail=1` to list failures from the
previous run. Inspect the generated scripts and logs under `$PETSC_ARCH/tests/`; the `test*err.log`
files provide the aggregate failure details. Reproduce the narrowest failing test rather than
rerunning a broad selection.

## Update expected output

Use `REPLACE=1` only when the new output is known to be correct. After replacement:

1. Inspect every changed `output/*.out` file.
2. Rerun the selected test without `REPLACE=1`.
3. Require a clean result under the test's specific `petscdiff` invocation.

Normal `petscdiff` use does not compare floating-point numbers unless the test has `-j` in
`diff_args` or `DIFF_NUMBERS=1` is requested.

Do not use `ALT=1` or generated-script `-M` to create or replace alternative output files. PETSc
tries to avoid new alternate outputs; leave any exceptional update to the user.

## petsc4py tests

Before running petsc4py tests, follow the build prerequisites and environment guidance in the
petsc4py section of [petsc-build](../petsc-build/SKILL.md#petsc4py).

The test runner automatically uses the Python build directory produced by `setup.py build`.
Select tests with one or more `-k` substring patterns:

```console
$ PETSC_ARCH=arch-name python test/runtests.py -k test_name
$ PETSC_ARCH=arch-name python test/runtests.py -k test_name.class_name.method_name
```

For behavior that depends on MPI communication, run the selected tests with the MPI launcher
for the supplied architecture and the relevant rank count. Check that the selection actually
runs the intended tests and inspect failures on every rank. CI jobs run tests serially or with
a configured MPI rank count, often four. Check the job's actual command and `PETSC4PY_NP` in
the generated configuration rather than assuming that every job uses four ranks.

Write tests to work with the relevant serial and MPI rank counts where the operation supports
them. If a test requires a particular rank count, skip it explicitly at unsupported counts and
report any resulting gap in coverage of the changed behavior.
