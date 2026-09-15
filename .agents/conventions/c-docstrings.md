# Docstring Conventions (`/*@ ... @*/`)

`petsclinter` enforces docstring formatting. See [petsc-lint](../skills/petsc-lint/SKILL.md) for commands and dependencies.

- **Section order.** Sections in `/*@ ... @*/` always appear in this order:
  1. One-line synopsis (`FunctionName - one-line description`)
  2. Collectivity (`Collective`, `Logically Collective`, `Not Collective`, `Asynchronous`)
  3. `Input Parameter(s):`
  4. `Output Parameter(s):`
  5. `Options Database Key(s):`
  6. `Level:`  ← always before Notes
  7. `Notes:` / `Note:`
  8. `Example Usage:`
  9. `Fortran Notes:`
  10. `.seealso:`

Two recurring traps the linter catches:

- **Param-list alignment.** In `Input Parameters:` / `Output Parameters:` blocks, every entry's `-` must sit exactly one space past the longest valid argument name. With args `da, xyz, bd, H` (longest is `xyz`), the correct form is:
  ```
  + da  - the `PetscDA` context
  . xyz - array of coordinate vectors
  . bd  - array of periodic-domain extents
  - H   - the observation operator
  ```
  Continuation lines for a multi-line description must be indented to line up under the description (i.e., the column right after `- `), not under the argument name.

- **Stray paragraphs in `Notes:`.** A colon-less line is flagged as a possible section header (`-fdoc-section-header-maybe-header`) only when it *begins with a section title* — `Note`, `Notes`, `Level`, `Collective`, `Input Parameter`, `Output Parameter`, `Options Database`, `Example Usage`, `Fortran Notes`, `Developer Note`, and the like. Ordinary prose starting with any other capitalized word ("When", "If", "The") is fine, blank line or not. If a paragraph does start with a section title, rephrase it or fold it into the preceding paragraph.

When in doubt, pattern-match against existing well-formatted docstrings in the same file.
