# Developing PETSc Documentation

```{toctree}
:maxdepth: 2
```

## General Guidelines

- Good documentation should be like a bonsai tree: alive, on display, frequently tended, and as small as possible (adapted from [these best practices](https://github.com/google/styleguide/blob/gh-pages/docguide/best_practices.md)).
- Wrong, irrelevant, or confusing documentation is worse than no documentation.

(sphinx_documentation)=

## Documentation with Sphinx

We use [Sphinx](https://www.sphinx-doc.org/en/master/) to build our web pages and documentation. Most content is written using [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html), a simple markup language.

[These slides](https://gitlab.com/psanan/petsc-sphinx-slides) contain an overview of Sphinx and how we use(d) it, as of October 2020.

(sec_local_html_docs)=

### Building the HTML docs locally

We use a [Python 3 virtual environment](https://docs.python.org/3/tutorial/venv.html) to build the documentation since not all developers can trivially install the needed Python modules directly.

```console
$ cd $PETSC_DIR
$ make docs
$ open doc/_build/html/index.html  # in a browser
```

or

```console
$ cd $PETSC_DIR/doc
$ make sphinxhtml
$ open _build/html/index.html
```

(sec_local_docs_latex)=

### Building the manual locally as a PDF via LaTeX

:::{admonition} Note
Before following these instructions, you need to have a working
local LaTeX installation and the ability to install additional packages,
if need be, to resolve LaTeX errors.
:::

Set up your local Python environment (e.g., ref:`as above <sec_local_html_docs>`), then

```console
$ cd doc
$ make sphinxpdf
$ open _build/latex/manual.pdf  # in PDF viewer
```

(sphinx_guidelines)=

### Sphinx Documentation Guidelines

Refer to Sphinx's [own documentation](https://https://www.sphinx-doc.org) for general information on how to use Sphinx, and note the following additional guidelines.

- Use the [literalinclude directive](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-literalinclude) to directly include pieces of source code. Use a path beginning with `/`, relative to the root for the Sphinx docs (where `conf.py` is found).

  ```rst
  .. literalinclude:: /../src/sys/error/err.c
     :start-at: PetscErrorCode PetscError(
     :end-at: PetscFunctionReturn(PETSC_SUCCESS)
     :append: }
  ```

  For robustness to changes in the source files, Use `:start-at:` and related options when possible, noting that you can also use (positive) values of `:lines:` relative to this. Use the `:language:` option to appropriately highlight languages other than C.

- Any invocable command line statements longer than a few words should be in
  `.. code-block::` sections. Double backticks must enclose any such statements not in code-block statements"\`\`". For example `make all` is acceptable but

  ```console
  $ make PETSC_DIR=/my/path/to/petsc PETSC_ARCH=my-petsc-arch all
  ```

  should be in a `.. code-block::`.

- All code blocks showing command line invocation must use the "console" block
  directive. E.g.

  ```rst
  .. code-block:: console

     $ cd $PETSC_DIR/src/snes/interface
     $ ./someprog
     output1
     output2
  ```

  The only exception to this is when displaying raw output, i.e., with no preceding
  commands. Then one may use just the "::" directive to improve visibility, e.g.,

  ```rst
  ::

     output1
     output2
  ```

- Any code blocks that show command line invocations must be preceded by `$`, e.g.

  ```rst
  .. code-block:: console

     $ ./configure --some-args
     $ make libs
     $ make ./ex1
     $ ./ex1 --some-args
  ```

- Environment variables such as `$PETSC_DIR` or `$PATH` must be preceded by
  `$` and be enclosed in double backticks, e.g.

  ```rst
  Set ``$PETSC_DIR`` and ``$PETSC_ARCH``
  ```

- For internal links, use explicit labels, e.g

  ```rst
  .. _sec_short_name:

  Section name
  ============
  ```

  and elsewhere (in any document),

  ```rst
  See :ref:`link text <sec_short_name>`
  ```

- For internal links in the manual with targets outside the manual, always provide alt text
  so that the text will be properly formatted in the {ref}`standalone PDF manual <sec_local_docs_latex>`, e.g.

  > ```rst
  > PETSc has :doc:`mailing lists </community/mailing>`.
  > ```

- We use the [sphinxcontrib-bibtex extension](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/)
  to include citations from BibTeX files.
  You must include `.. bibliography::` blocks at the bottom of a page, including citations ([example](https://gitlab.com/petsc/petsc/-/raw/main/doc/manual/ksp.rst)).
  To cite the same reference on more than one page, use [this workaround](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#key-prefixing) on one of them ([example](https://gitlab.com/petsc/petsc/-/raw/main/doc/developers/articles.rst)) [^bibtex-footnote].

- See special instructions on {any}`docs_images`.

- Prefer formatting styles that are easy to modify and maintain. In particular, the use of [list-table](https://docutils.sourceforge.io/docs/ref/rst/directives.html#list-table) is recommended.

- When using external links with inline URLs, prefer to use [anonymous hyperlink references](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#anonymous-hyperlinks) with two trailing underscores, e.g.

  ```rst
  `link text <https://external.org>`__
  ```

- To pluralize something with inline markup, e.g. `DM`s, escape the trailing character to avoid `WARNING: Inline literal start-string without end-string`.

  ```rst
  ``DM``\s
  ```

- Use restraint in adding new Sphinx extensions, in particular, those which aren't
  widely used and well-supported, or those with hidden system dependencies.

(petsc_repositories)=

## Other PETSc repositories

In addition to the [PETSc repository](https://gitlab.com/petsc/petsc), there are three other PETSc repositories which contain large data
files that are unnecessary for most PETSc usages and thus are not stored in the main repository.

- [Images](https://gitlab.com/petsc/images) contains images that are used in the PETSc documentation or have other uses. See {any}`docs_images`
  for details on its use.
- [Annual-Meetings](https://gitlab.com/petsc/annual-meetings) contains various documents from the {any}`meetings`. See {any}`docs_meetings`.
- [Datafiles](https://gitlab.com/petsc/datafiles) contains large matrices, meshes, and various other data files that
  are used in the {any}`PETSc CI<test_harness_data>`.
- [Tutorials]((https://gitlab.com/petsc/annual-meetings) contains slides from {any}`tutorials`. See {any}`docs_tutorials`.

Other repositories containing software PETSc uses are located at [GitLab](https://gitlab.com/petsc/)
and [BitBucket](https://bitbucket.org/petsc/workspace/repositories). The BitBucket location is used for historical reasons,
there are many links on the web to these locations thus the repositories have not been migrated to GitLab.

(docs_images)=

## Images

PETSc's documentation is tightly coupled to the source code and tests and
is tracked in the primary PETSc Git repository. However, image files are
too large to track directly this way (especially because they persist in the integration branches' histories). Thus we do not put images
into the PETSc git repository.

Therefore, we store image files in a separate Git repository, [Images](https://gitlab.com/petsc/petsc). This repository is automatically cloned
(if not already available) and updated  when building the documentation. It can also be cloned by running
`make images` in the `doc/` directory.
Any new images required must be added to the currently-used branch of this repository.

### Image Guidelines

- Whenever possible, use SVG files. SVG is a web-friendly vector format and will be automatically converted to PDF using `rsvg-convert` [^svg-footnote]
- Avoid large files and large numbers of images.
- Do not add movies or other non-image files.

### Adding new images

- Decide where in `doc/images` a new image should go. Use the structure of the `doc/` tree as a guide.
- Create a Merge Request to the currently-used branch of the upstream images repository, adding this image [^maintainer-fast-image-footnote].
- Once this Merge Request is merged, you may make a MR on the main PETSc Git repository relying on the new image(s).

It may be helpful to place working copies of the new image(s) in your local `doc/images`
while iterating on documentation; don't forget to update the upstream images repository.

### Removing, renaming, moving, or updating images

Do not directly move, rename, or update images in the images repository.
Simply add a logically-numbered new version of the image.

If an image is not used in *any* {any}`integration branch <sec_integration_branches>` (`main` or `release`),
add it to the top-level list of files to delete in the images repository.

(docs_images_cleanup)=

### Cleaning up the images repository (maintainers only)

If the size of the image repository grows too large,

- Create a new branch `main-X`, where `X` increments the current value
- Create a new commit deleting all files in the to-delete list and clearing the list
- Reset the new `main-X` to a single commit with this new, cleaned-up state
- Set `main-X` as the "default" branch on GitLab.
- Update both `release` and `main` in the primary PETSc repository to clone this new branch

(docs_meetings)=

## Annual meetings website

Like {any}`docs_images` the material (slides, etc.) for the PETSc annual meetings is too large to store in the primary PETSc Git repository.
It is stored in [Annual-Meetings](https://gitlab.com/petsc/annual-meetings) repository and linked from {any}`meetings`.

The files are all in the public directory of the repository so that the `.gitlab-ci.yml` file for the repository
automatically displays all the files at https://petsc.gitlab.io/annual-meetings. Thus, all one needs to do is add files into
[Annual-Meetings](https://gitlab.com/petsc/annual-meetings) and provide appropriate links within that repository or from {any}`meetings`
in the primary PETSc Git repository.

(docs_tutorials)=

## Tutorials website

Like {any}`docs_meetings` the material (slides, etc.) for the PETSc tutorials is too large to store in the primary PETSc Git repository.
It is stored in [Tutorials](https://gitlab.com/petsc/tutorials) repository and linked from {any}`tutorials`.

The files are all in the public directory of the repository so that the `.gitlab-ci.yml` file for the repository
automatically displays all the files at https://petsc.gitlab.io/tutorials. Thus, all one needs to do is add files into
[Tutorials](https://gitlab.com/petsc/tutorials) and provide appropriate links within that repository or from {any}`tutorials`
in the primary PETSc Git repository.

(manpages_c2html_build)=

## Building Manual Pages and C2HTML Files

The manual pages and C2HTML-generated file as built in a process described below using the documentation tools listed below, which are
automatically downloaded and installed if needed while building the PETSc documentation./

- [Sowing](https://bitbucket.org/petsc/pkg-sowing): Developed by Bill Gropp, this produces the PETSc manual pages; see the [Sowing documentation](http://wgropp.cs.illinois.edu/projects/software/sowing/doctext/doctext.htm) and {ref}`manual_page_format`.
- [C2html](https://gitlab.com/petsc/pkg-c2html): This generates the HTML versions of all the source code.

Sowing and C2html are build tools that do not use the compilers specified to PETSc's `configure`, as they
need to work in cross-compilation environments. Thus, they default to using `gcc`, `g++`, and `flex` from
the user's environment (or `configure` options like `--download-sowing-cxx`). Microsoft Windows users must install `gcc`
etc., from Cygwin in order to be able to build the documentation.

```{rubric} Footnotes
```

[^bibtex-footnote]: The extensions's [development branch](https://github.com/mcmtroffaes/sphinxcontrib-bibtex) [supports our use case better](https://github.com/mcmtroffaes/sphinxcontrib-bibtex/pull/185) (`:footcite:`), which can be investigated if a release is ever made. This stuff is now in the main repository but does not work as advertised from .md files.

[^svg-footnote]: `rsvg-convert` is installable with your package manager, e.g., `librsvg2-bin` on Debian/Ubuntu systems).

[^maintainer-fast-image-footnote]: Maintainers may directly push commits.
