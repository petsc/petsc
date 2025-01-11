(ch_developingmr)=

# Developing a Merge Request

(sec_integration_branches)=

## Select the integration branch

**Integration branches** are permanent branches in a repository that developers can contribute to. PETSc has two integration branches: `release`
and `main`. **Feature branches** are temporary branches created by developers to add or change a feature. A new feature branch is the basis for each
merge request.

(sec_release_branch)=

### `release`

The `release` branch contains the latest PETSc release including bug-fixes.

Bug-fixes, along with most documentation fixes, should start from `release`.

```console
$ git fetch
$ git checkout -b yourname/fix-component-name origin/release
```

Bug-fix updates, about every month, (e.g. 3.17.1) are tagged on `release` (e.g. v3.17.1).

(sec_main_branch)=

### `main`

The `main` branch contains everything in the release branch as well as new features that have passed all testing
and will be in the next release (e.g. version 3.18). Users developing software based
on recently-added features in PETSc should follow `main`.

New features should start from `main`.

```console
$ git fetch
$ git checkout -b yourname/fix-component-name origin/main
```

(sec_developing_a_new_feature)=

## Start a new feature branch

- Determine the appropriate integration_branch to start from, `main` or `release` (for documentation and bug fixes only).

- Create and switch to a new feature branch:

  ```console
  $ git fetch
  $ git checkout -b <loginname>/<affected-package>-<short-description> origin/main  # or origin/release
  ```

  For example, Barry’s new feature branch on removing CPP in `snes/` will
  use

  ```console
  $ git checkout -b barry/snes-removecpp origin/main
  ```

  Use all lowercase and no additional underscores in the branch name.

## Develop your code

- Write code and tests.

- For any new features or API changes you introduced add information on them to `doc/changes/dev.rst`.

- Inspect changes and stage code using standard Git commands, e.g.

  ```console
  $ git status
  $ git add file1 file2
  $ git commit
  ```

- Commit code with good commit message, for example

  ```console
  $ git commit
  ```

  ```none
  ComponentName: one-line explanation of commit

  After a blank line, write a more detailed explanation of the commit. Many tools do not auto-wrap this part, so wrap paragraph text at a reasonable length. Commit messages are meant for other people to read, possibly months or years later, so describe the rationale for the change in a manner that will make sense later, and which will be provide helpful search terms.

  Use the imperative, e.g. "Fix bug", not "Fixed bug".

  If any interfaces have changed, the commit should fix occurrences in PETSc itself and the message should state its impact on users.

  We have defined several standard commit message tags you should use; this makes it easy to search for specific types of contributions. Multiple tags may be used in the same commit message.

  /spend 1h or 30m

  If other people contributed significantly to a commit, perhaps by reporting bugs or by writing an initial version of the patch, acknowledge them using tags at the end of the commit message.

  Reported-by: Helpful User <helpful@example.com>
  Based-on-patch-by: Original Idea <original@example.com>
  Thanks-to: Incremental Improver <improver@example.com>

  If work is done for a particular well defined funding source or project you should label the commit with one or more of the tags

  Funded-by: My funding source
  Project: My project name
  ```

- Push the feature branch to the remote repository as desired:

  ```console
  % git push -u origin barry/snes-removecpp
  ```

## Test your branch

- Include {doc}`tests </developers/testing>` which cover any changes to the source code.

- {any}`Run the full test suite <sec_runningtests>` on your machine.

  ```console
  $ make alltests TIMEOUT=600
  ```

- Run the source checkers on your machine.

  ```console
  $ make checkbadSource
  $ make clangformat
  $ make lint
  ```

(sec_clean_commit_history)=

## Maintain a clean commit history

If your contribution can be logically decomposed into 2 or more
separate contributions, submit them in sequence with different
branches and merge requests instead of all at once.

Often a branch's commit history does not present a logical series of changes.
Extra commits from bug-fixes or tiny improvements may accumulate. One commit may contain multiple orthogonal changes.
The order of changes may be incorrect. Branches without a clean commit history will often break `git bisect`.
Ideally, each commit in an MR will pass the PETSc CI testing, while presenting a small-as-possible set of very closely related changes.

Use different commits for:

- fixing formatting and spelling mistakes,
- fixing a bug,
- adding a new feature,
- adding another new feature.

Rewriting history can be done in [several ways](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History); the easiest is often with the interactive `rebase` command, which allows one to combine ("squash"), rearrange, and edit commits.

It is better to clean up your commits regularly than to wait until you have a large number of them.

For example, if you have made three commits and the most recent two are fixes for the first, you could use

```console
$ git rebase -i HEAD~3
```

If the branch has already been pushed, the rewritten branch is not compatible with the remote copy of the branch. You must force push your changes with

```console
$ git push -f origin branch-name
```

to update the remote branch with your copy. This must be done with extreme care and only if you know someone else has not changed the remote copy of the branch,
otherwise you will lose those changes. Never do a `git pull` immediately after you rebase since that will merge the old branch (from GitLab) into your local one and create a mess [^block-ugly-pull-merge].

You can use `git log` to see the recent changes to your branch and help determine what commits should be rearranged, combined, or split.
You may also find it helpful to use an additional tool such as
[git-gui](https://git-scm.com/docs/git-gui/), [lazygit](https://github.com/jesseduffield/lazygit), or [various GUI tools](https://git-scm.com/downloads/guis).

(sec_rebasing)=

## Rebase your branch against the integration branch

You may also need to occasionally [rebase](https://git-scm.com/book/en/v2/Git-Branching-Rebasing) your branch onto to the latest version of your {any}`integration branch <sec_integration_branches>` [^rebase-not-merge-upstream], if the integration branch has had relevant changes since you started working on your feature branch.

```console
$ git fetch origin                              # assume origin --> PETSc upstream
$ git checkout myname/component-feature
$ git branch myname/component-feature-backup-1  # optional
$ git rebase origin/main                        # or origin/release
```

Note that this type of rebasing is different than the `rebase -i` process for organizing your commits in a coherent manner.

```{rubric} Footnotes
```

[^rebase-not-merge-upstream]: Rebasing is generally preferable to [merging an upstream branch](http://yarchive.net/comp/linux/git_merges_from_upstream.html).

[^block-ugly-pull-merge]: You may wish to [make it impossible to perform these usually-undesired "non fast-forward" merges when pulling](https://git-scm.com/docs/git-config#Documentation/git-config.txt-pullff), with `git config --global pull.ff only`.
