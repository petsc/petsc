# PETSc Documentation Conventions

- Add `doc/changes/dev.md` entries for new features or API changes, otherwise only when requested. Preserve pre-existing entries; all other Markdown files in `doc/changes/` are immutable.
- Describe designs conceptually. For copied PETSc source beyond standalone prototypes, use `literalinclude` with narrow `:start-at:` and `:end-at:` anchors. Keep illustrative code, pseudocode, and standalone prototypes in fenced blocks; retain prototype parameter names referenced by the text.
- Mention nonexistent, obsolete, or deprecated symbols only for historical, migration, or compatibility context, never as current API.
