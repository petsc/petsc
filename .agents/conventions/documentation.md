# PETSc Documentation Conventions

- Changelog entries. Apply these rules to PETSc and to petsc4py:
  - Add each PETSc entry to `doc/changes/dev.md`.
  - Add each petsc4py entry to the `Development` section of `src/binding/petsc4py/CHANGES.rst`.
  - If a change affects both PETSc and petsc4py, add one entry to each file.
  - Add an entry for each of these changes:
    - A new public API.
    - A change to an existing public API or to its documented behavior.
    - A major change to configuration, build, or installation. Examples: a new or removed configure option, a changed configure default, or a new minimum version of an external package.
  - Do not add an entry for a bug fix, unless the fix also matches an item in the list above or the user asks for one.
  - Do not add an entry for any other change, unless the user asks for one.
  - Keep each entry brief. State what changed. If users must change their code or build, state what they must do.
  - Put details in the manual pages and other documentation, not in the entry.
  - Do not change or remove an entry that the current branch did not add.
  - In `doc/changes/`, edit only `dev.md`. In `CHANGES.rst`, edit only the `Development` section.
- Describe designs conceptually. For copied PETSc source beyond standalone prototypes, use `literalinclude` with narrow `:start-at:` and `:end-at:` anchors. Keep illustrative code, pseudocode, and standalone prototypes in fenced blocks; retain prototype parameter names referenced by the text.
- Mention nonexistent, obsolete, or deprecated symbols only for historical, migration, or compatibility context, never as current API.
