(doc_creepycrawly)=

# Reporting Bugs, Asking Questions, and Making Suggestions

Use any of

- <mailto:petsc-users@mcs.anl.gov> (public mailing list with public archives)
- <mailto:petsc-maint@mcs.anl.gov> (private maintenance mailing list without archives). Installation
  issues generally require sending the files `configure.log` and `make.log`, i.e., uncompressed large
  attachments, <mailto:petsc-maint@mcs.anl.gov> is preferable.
- [PETSc on Discord](https://discord.gg/Fqm8r6Gcyb)
- [PETSc GitLab Issues](https://gitlab.com/petsc/petsc/-/issues)

Topics can include:

- Report bugs.
- Ask for clarification.
- Ask for help in tracking down bugs.
- Request new features within PETSc.
- Recommend changes or additions to the development team.

:::{important}
Please `do not send email requests to the individual PETSc authors`; all list email
is automatically distributed to all of the PETSc authors, so our response time here
will be fastest.
:::

Before sending a bug report, please consult the {ref}`FAQ <doc_config_faq>` to determine
whether a fix or work-around to the problem already exists. Also, see the chapter on
{ref}`performance tuning <ch_performance>` in the PETSc users manual for guidelines on
achieving good efficiency within PETSc codes.

(sec_doc_fixes)=

## Small Documentation fixes

We welcome corrections to our documentation directly by clicking "Edit this page", on the upper right corner of the page,
making your edits, and following the instructions to make a merge request. Merge requests for such fixes should always have the GitLab `docs-only` label set.

## Guidelines For Bug Reports

The more information that you convey about a bug, the easier it will be for us to target
the problem. We suggest providing the following information:

:::{admonition} Don'ts
:class: yellow

- Please do **not** send winmail.dat Microsoft email attachments.
- Please do **not** send screenshots, use cut-and-paste from terminal windows to send text.
- Please do **not** put huge files like `configure.log` DIRECTLY into the email
  message. Instead, include them as attachments.
- Please do NOT paste **entire** programs DIRECTLY into the email message. Instead,
  include them as attachments. Small snippets of code in the messages are acceptable however.
:::

:::{admonition} Do's
- Detailed steps to recreate the problem if possible.

- Copy of the **complete** error message using cut-and-paste, if feasible, otherwise include the full error
  message as a **text** attachment, not a screenshot.

- If the problem involves installation, send the entire `configure.log` and
  `make.log` files as attachments.

  - `configure.log` can be found either at `$PETSC_DIR/configure.log`,
    `$PETSC_DIR/configure.log.bkp` (which holds the second-most recent
    `configure.log`), or in
    `$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/configure.log[.bkp]`.
  - `make.log` can be found in the same places as listed above, however note that
    there is no `make.log.bkp` so be sure to not overwrite your `make.log` with
    additional build attempts.

- Machine type: HPC, laptop, etc.

- OS version and type: run `uname -a` to get the version number

- PETSc version: run any PETSc program with the additional command-line option `-version`, or look in
  `$PETSC_DIR/include/petscversion.h`

- MPI implementation: MPICH, Open MPI, IBM, Intel, etc.

- Compiler and version: GNU, Clang, Intel, etc.

- Probable PETSc component: `Mat`, `Vec`, `DM`, `KSP`, etc.
:::

(doc_mail)=

# Mailing Lists

The following mailing lists, with public archives, are available.

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Name
     - Purpose
     - Subscribe
     - Archive
   * - petsc-announce@mcs.anl.gov
     - For announcements regarding PETSc :ref:`releases <doc_download>`
     - `subscribe/unsubscribe <https://lists.mcs.anl.gov/mailman/listinfo/petsc-announce>`__
     - `archives <http://lists.mcs.anl.gov/pipermail/petsc-announce/>`__
   * - petsc-users@mcs.anl.gov
     - For PETSc users
     - `subscribe/unsubscribe <https://lists.mcs.anl.gov/mailman/listinfo/petsc-users>`__
     - `archives <http://lists.mcs.anl.gov/pipermail/petsc-users/>`__
   * - petsc-dev@mcs.anl.gov
     - For PETSc developers and others interested in the development process
     - `subscribe/unsubscribe <https://lists.mcs.anl.gov/mailman/listinfo/petsc-dev>`__
     - `archives <http://lists.mcs.anl.gov/pipermail/petsc-dev/>`__
```

:::{important}
<mailto:petsc-maint@mcs.anl.gov> - a private maintenance e-mail without public archives - is
also available. Send issues requiring large attachments here, in particular
uncompressed `configure.log` and `make.log` when encountering installation
issues.

Also see {ref}`doc_creepycrawly`.
:::

:::{note}
- petsc-announce is an announcement-only list (users cannot post).
- petsc-users and petsc-dev are open; we recommend subscribing and participating
  in the list discussions. However, it is possible to post to the lists without
  subscribing (the first post to the list will be held until list owner can
  enable access)
- Avoid cross posting to multiple lists. You can reach PETSc developers equally
  on any suitable list.
:::
