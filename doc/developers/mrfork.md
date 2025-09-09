(mr_fork)=

# Checkout fork merge request branch

`Developers` at times, need to checkout and build changes from a merge request fork branch. Any one of the following methods can be used:

- Checkout `COMMIT-SHA` of the branch HEAD. It is available on the "Commits" tab of the merge request web page.

  ```console
  % git fetch origin <COMMIT-SHA>
  % git checkout FETCH_HEAD
  ```

- Checkout branch using the repository `URL`. The `URL with branchname` is available as "Source branch" hyperlink on merge request web page.

  ```console
  % git fetch <URL> <branchname>
  % git checkout FETCH_HEAD
  ```

- Setup a local Git clone to access the merge request branch via the `MR-NUMBER`. Here use the following in `.git/config`:

  ```console
  [remote "origin"]
        url = https://gitlab.com/petsc/petsc.git
        pushurl = git@gitlab.com:petsc/petsc.git
        fetch = +refs/heads/*:refs/remotes/origin/*
        fetch = +refs/merge-requests/*:refs/remotes/origin/merge-requests/*
  ```

  Now, the branch is available to checkout:

  ```console
  % git fetch
  % git checkout origin/merge-requests/<MR-NUMBER>/head
  ```

# Commit and push changes to a merge request fork branch

Only `Owners/Maintainers` can push commits to a merge request fork branch. Here, use the `ssh-URL` for the Git repository.

```console
% git fetch <ssh-URL> <branchname>
% git checkout -b <branchname> FETCH_HEAD
% git push -u <ssh-URL> <branchname>
% (edit/commit)
% git push
```

Notes:

For example, with `merge request` at <https://gitlab.com/petsc/petsc/-/merge_requests/8619> we have:

- `MR-NUMBER` = `8619`
- `COMMIT-SHA` = `1b741c341f10772b6231f15a4abcef052bfe2d90` ("Copy commit SHA" of the HEAD commit from the "Commits" tab)
- `URL with branchname` = `https://gitlab.com/paul.kuehner/petsc/-/tree/add-tao-get-constraints` ("Copy link" of "Source branch")
- `URL` = `https://gitlab.com/paul.kuehner/petsc`
- `ssh-URL` = `git@gitlab.com:paul.kuehner/petsc.git`
- `branchname` = `add-tao-get-constraints`

