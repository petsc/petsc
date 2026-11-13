---
name: clean-commits
description: >-
  Restructure a PETSc branch into an ordered, incremental commit series and verify each commit
  builds and passes its checks. Use when asked to clean up commits, reorder, split, or squash them,
  rewrite commit messages, or make a branch ready for a merge request.
argument-hint: <upstream ref | empty to detect origin/main or origin/release>
---

# Clean a PETSc branch's commit history

- Produce a series a reviewer reads top to bottom.
- Make each commit small, self-contained, and passing CI on its own.
- This skill implements `sec_clean_commit_history` in `doc/developers/contributing/developingmr.md`.
- Run from the repository root.
- `arch-name` is `arch-check-commits` unless the user supplies another. It is isolated from the
  working `PETSC_ARCH`.
- Do not push. `git push -f` overwrites the remote branch.

## 1. Establish the baseline

- Stop if `git status --porcelain --untracked-files=no` prints anything. Ask the user to commit
  or stash those changes. `git reset BASE` and `git checkout -- .` would commit or discard them.
- `UPSTREAM` is `$ARGUMENTS`, or else the output of `lib/petsc/bin/maint/check-merge-branch.sh`.
- `BASE` is `git merge-base UPSTREAM HEAD`.
- Keep `BASE` as parent. Do not rebase onto a newer upstream.
- If the series already meets Sections 2 and 5, say so, rewrite nothing, and ask only about
  verification (Section 2).

```console
$ git log --stat BASE..HEAD          # subjects, authors, trailers, touched files
$ git rev-parse HEAD                 # record as OLD
```

- Preserve author, trailers, and any message content not reconstructable from the diff.

## 2. Choose squash or series, then order the commits

- Decide if one squashed commit suffices. Use one for a single logical change of reviewable size.
- Otherwise build a series in this order:
  1. Fixes and re-formats of existing code.
  2. Changes to existing code that the new feature needs.
  3. The new features, split as far as they decompose naturally.
- Prefer incremental commits. Do not over-split.
- Group related changes (e.g., same fix in two functions with its test).
- Keep orthogonal changes separate: no formatting hunks with logic.
- A test added by commit N must not exercise behavior that arrives in commit N+1. Splitting a
  feature across commits splits its test block and expected output too.

Before rewriting, ask the user to approve a plan and wait for the answer:

- Derive the plan from this branch's changes and the rules above. List its commits in order with
  subject and files. Offer an alternative only where the rules allow more than one choice.
- In the same prompt, propose checks for each commit from what it touches (Sections 6 and 7).
  Let the user change them.

## 3. Rebuild history from the final tree

Rebuild from the final tree. Do not replay the old commits with `git rebase -i`:

```console
$ git branch <branch>-backup         # cheap undo
$ git reset BASE                     # final tree in working tree, nothing staged, parent kept
$ git add <paths>                    # one commit's files
$ git commit
```

- `<branch>` is the output of `git branch --show-current`. If `<branch>-backup` exists, pick an
  unused name (for example `<branch>-backup2`). Never overwrite an earlier backup.
- To split one file across commits, commit its intermediate state, then restore the final state
  with `git restore --source=OLD -- <file>`. Do not use `git checkout OLD -- <file>`, which also
  stages the file.

Then prove nothing was lost or gained:

```console
$ git diff OLD HEAD                  # empty unless step 3 below changed the final tree
```

- No later commit fixes an earlier one. When a commit fails verification, try these in order:
  1. Move existing hunks between commits, or reorder commits.
  2. Give the failing commit an intermediate state that a later commit replaces.
  3. Change the final tree, only if the same failure occurs at `OLD`. Report that diff.
- Steps 1 and 2 must leave `git diff OLD HEAD` empty. Fold each change into the commit of the new
  series that introduced the problem, then re-verify:

  ```console
  $ git commit --fixup=<sha>
  $ GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash BASE
  ```

## 4. Preserve authorship

- Keep the original author (`--author='Name <email>'`) and all trailers through splits and
  squashes. Change an LLM `Co-authored-by:` to `Assisted-by:`, as `AGENTS.md` requires.
- Check that change with `python3 lib/petsc/bin/maint/check_commit_trailers.py BASE..HEAD`.
  The script does not check authors or other trailers.

## 5. Write the messages

```none
Component: imperative one-line summary

- significant change
- significant change
```

- Subject: component and imperative ("Fix bug", not "Fixed bug"). Omit body when subject suffices.
- Body: bullets for significant changes only, one per user-visible change or per file, not per
  edit. Reuse original wording. Keep log blocks verbatim. State impact of interface changes.
- Capitalize "Git". Commands in lowercase.
- Follow `AGENTS.md` writing rules.

## 6. Verify commits

- Verify each commit with the checks the user approved in Section 2.
- For a commit that needs a build, check it out and run the chain below. Configure once.
  Re-run configure only for `configure` or `config/` changes. See
  [petsc-configure](../petsc-configure/SKILL.md) and [petsc-build](../petsc-build/SKILL.md).

```console
$ ./configure PETSC_ARCH=arch-name
$ make PETSC_ARCH=arch-name fortranbindings     # after every checkout, arch-name/ftn/** is stale
$ make PETSC_ARCH=arch-name all
$ make PETSC_ARCH=arch-name check               # smoke test only, see Section 7
$ make PETSC_ARCH=arch-name checkbadSource
$ make PETSC_ARCH=arch-name checkclangformat
```

- Run `make PETSC_ARCH=arch-name cleantest` between checkouts to remove stale test artifacts
  (`$PETSC_ARCH/tests/` and generated test files).
- Verify in order in one tree (incremental builds). For many compiled-code commits, use up to 3
  sub-agents in parallel, each in its own worktree
  (`git worktree add --detach ../petsc-check-N <commit>`) with its own `arch-name`. Remove the
  worktrees when done.
- `checkclangformat` reformats files in place and fails if any file changed. Handle the failure as
  Section 3 describes, then run `git checkout -- .` before the next checkout. See
  [petsc-lint](../petsc-lint/SKILL.md).

## 7. Extra verification the commit's content demands

For a commit where the user approved tests, add the checks for each trigger:

- External package interface changed, or a changed test needs one: add `--download-<package>` and
  its dependencies to the `arch-name` configure, and report them. Run the chain, then only the
  package's tests that exercise the commit's changes. Intersect the package requirement with a
  test-name glob from the changed directories, or an argument glob for the options that select the
  changed code. `query='requires|name'` with `queryval='hypre,glob'` selects tests that match
  both.

  ```console
  $ make PETSC_ARCH=arch-name print-test query='requires|name' queryval='hypre,mat_tests*'
  $ make PETSC_ARCH=arch-name print-test query='requires|args' queryval='hypre,*pc_type*hypre*'
  $ make PETSC_ARCH=arch-name TIMEOUT=600 test query='requires|args' queryval='hypre,*pc_type*hypre*'
  ```

  Run all the package's tests (`query='requires' queryval='<package>'`) only when the commit
  changes how PETSc configures or downloads the package
  (`config/BuildSystem/config/packages/<package>.py`).
- petsc4py changed: add `--with-petsc4py=1` to the `arch-name` configure. Run the chain
  (`make all` builds petsc4py), then
  `PETSC_ARCH=arch-name mpiexec -n 4 python src/binding/petsc4py/test/runtests.py -k <pattern>`,
  for example `-k test_mat_dense`. Check the result as
  [petsc-test](../petsc-test/SKILL.md#petsc4py-tests) describes.
- Tests or tutorials changed: run the selection in `arch-name`. Several failure modes look like a
  pass, so read the result as [petsc-test](../petsc-test/SKILL.md#read-the-result) describes.
  Always pass `TIMEOUT=600` to `make test`:

  ```console
  $ make PETSC_ARCH=arch-name TIMEOUT=600 test search='*pattern*'   # a glob over the test path
  $ make PETSC_ARCH=arch-name TIMEOUT=600 test search='*mat*tests*ex254*'   # example
  ```

## 8. Report

- Per commit: subject, what moved, and verification result with test selection.
- Squash or series, and why.
- What was reordered, split, squashed, or dropped.
- Tree vs backup: identical, or the `git diff OLD HEAD` output and the commit that carries each fix.
- The backup name, the undo command `git reset --hard <branch>-backup`, and the cleanup command
  `git branch -D <branch>-backup` for after the MR merges.
