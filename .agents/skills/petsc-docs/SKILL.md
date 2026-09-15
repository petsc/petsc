---
name: petsc-docs
description: >-
  Audit, edit, build, and diagnose PETSc or petsc4py documentation. Use for Markdown and
  Sphinx documentation work, documentation build failures, and an explicitly requested
  website, manual PDF, or petsc4py documentation check. An audit or edit does not itself
  authorize a documentation build.
---

# PETSc and petsc4py documentation

For an audit or edit, inspect the relevant Markdown or reStructuredText, toctrees, references,
docstrings, and included source.
Run a documentation build only when the user requests or approves it; reuse authorization
already given for the task.

## PETSc website and manual

When choosing a build command, inspect only the relevant targets and path variables in
`doc/makefile`, HTML builder hooks in `doc/conf.py`, and architecture selection in
`doc/build_manpages_c2html.py`. Consult the relevant preparation logic in `doc/prepare_docs.py`
only if the makefile uses it and its behavior needs clarification. Reuse these details while
the inspected sections are unchanged and remain in context; do not reread them for each build.
Expand the reads when a failure or unresolved dependency requires more context. These sections
determine the build directory, architecture, prerequisites, and generated output; do not assume
that all branches use the same layout.

- When the rules define `DOCS_DIR` under `PETSC_ARCH/doc/`, use their `SOURCEDIR` and `BUILDDIR`
  paths. Edit the original files under `doc/`, not the prepared source tree.
- Check whether the HTML hooks honor a supplied architecture or configure a fixed `arch-docs`.
  A directory named after `PETSC_ARCH` does not by itself mean the hooks use that configuration.
  If a requested architecture is unsupported, explain the limitation before running the build.
- For a supported user-supplied architecture, use its existing configuration and pass
  `PETSC_ARCH=arch-name` to make. Confirm that it provides `c2html` and Sowing's `doctext` and
  `mapnames` through its configuration or `PATH`. Report missing capabilities and follow
  [petsc-configure](../petsc-configure/SKILL.md) if configuration changes are requested.
- If the user requests the target's default documentation build, use the target's dedicated
  `arch-docs` configuration. An inherited environment value must not silently select a different
  architecture. This default build is the architecture exception in `AGENTS.md`.

Run the authorized HTML build from the repository root. For the dedicated documentation
architecture, the command is:

```console
$ make PETSC_ARCH=arch-docs docs
```

The HTML workflow prepares the Python documentation environment, generates manual pages and
C2HTML sources, builds PETSc and petsc4py, and renders the website. It can download tools and
documentation dependencies and clean generated files. When the user requests the manual PDF,
run `docspdf` after the HTML build, using the same architecture and the checkout's makefile rules.

HTML builds through `make -C doc html`, `website-deploy`, or direct Sphinx invocations also run
the HTML hooks. Do not use them as an implicit lightweight check during an audit. PDF and
link-check builders have different hooks; inspect those before describing their cost or side
effects. They also require a request or approval to run.

## Standalone petsc4py documentation

Apply the same scoped reading and reuse rules to the relevant documentation targets in
`src/binding/petsc4py/makefile` and binding imports in `docs/source/conf.py` under that directory.
These docs import the bindings for introspection, so use the architecture supplied by the user
and ensure its PETSc libraries are current as described in [petsc-build](../petsc-build/SKILL.md).

For an explicitly requested documentation check, run from `src/binding/petsc4py/`:

```console
$ make PETSC_ARCH=arch-name check-docs
```

This target recreates `petsc4py-docs-env`, installs the documentation requirements and the
bindings with C optimization disabled, and runs Sphinx with warnings treated as errors. Do not
add a separate bindings build when this target already supplies it. `make check-docs` is not a
mandatory completion step for every petsc4py change; it is useful when the MR's `docs-review`
job reports a problem. Run a direct `sphinx-html` target only with a prepared environment that
imports the intended bindings.

## Check the result

After a build, confirm from the output that Sphinx completed successfully and inspect warnings.
Do not rely solely on the wrapper's exit status: older shell recipes can mask intermediate
failures. Inspect the affected generated pages and links. Give the actual output path when the
user needs to inspect the result. Report relevant results and limitations, with commands and
architecture details when needed to reproduce or diagnose them. For an audit, report the findings
and any material limits of source inspection.
