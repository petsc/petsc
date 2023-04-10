==============================
Developing PETSc Documentation
==============================

.. toctree::
   :maxdepth: 2


General Guidelines
==================

* Good documentation should be like a bonsai tree: alive, on display, frequently tended, and as small as possible (adapted from `these best practices <https://github.com/google/styleguide/blob/gh-pages/docguide/best_practices.md>`__).
* Wrong, irrelevant, or confusing documentation is worse than no documentation.


.. _sphinx_documentation:

Documentation with Sphinx
=========================

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`__ to build our web page and documentation.  Most content is written using `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__, a simple markup language.

`These slides <https://gitlab.com/psanan/petsc-sphinx-slides>`__ contain an overview of Sphinx and how we use(d) it, as of October, 2020.


.. _sec_local_html_docs:

Building the HTML docs locally
------------------------------

We use a a `Python 3 virtual environment <https://docs.python.org/3/tutorial/venv.html>`__  [#venv_footnote]_ to build the documentation since not all developers can trivially install the needed Python modules directly.

.. code-block:: console

   $ cd $PETSC_DIR
   $ make docs

or

.. code-block:: console

   $ cd $PETSC_DIR/doc
   $ make sphinxhtml
   $ open _build/html/index.html

.. _sec_local_docs_latex:

Building the manual locally as a PDF via LaTeX
----------------------------------------------

.. admonition:: Note

   Before following these instructions, you need to have a working
   local LaTeX installation and the ability to install additional packages,
   if need be, to resolve LaTeX errors.

Set up your local Python environment (e.g. :ref:`as above <sec_local_html_docs>`), then

.. code-block:: console

   $ cd doc
   $ make sphinxpdf
   $ open _build/latex/manual.pdf  # or otherwise open in PDF viewer

.. _sphinx_guidelines:

Sphinx Documentation Guidelines
-------------------------------

Refer to Sphinx's `own documentation <https://https://www.sphinx-doc.org>`__ for general information on how to use Sphinx, and note the following additional guidelines.

* Use the `literalinclude directive <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-literalinclude>`__ to directly include pieces of source code. Use an "absolute" path, beginning with ``/``, which means relative to the root for the Sphinx docs (where ``conf.py`` is found).

  .. code-block:: rst

      .. literalinclude:: /../src/sys/error/err.c
         :start-at: PetscErrorCode PetscError(
         :end-at: PetscFunctionReturn(PETSC_SUCCESS)
         :append: }

  For robustness to changes in the source files, Use ``:start-at:`` and related options when possible, noting that you can also use (positive) values of ``:lines:`` relative to this. For languages other than C, use the ``:language:`` option to appropriately highlight.

* Any invocable command line statements longer than a few words should be in
  ``.. code-block::`` sections. Any such statements not in code-block statements must be
  enclosed by double backticks "``". For example ``make all`` is acceptable but

  .. code-block:: console

     $ make PETSC_DIR=/my/path/to/petsc PETSC_ARCH=my-petsc-arch all

  should be in a block.

* All code blocks showing invocation of command line must use the "console" block
  directive. E.g.

  .. code-block:: rst

     .. code-block:: console

        $ cd $PETSC_DIR/src/snes/interface
        $ ./someprog
        output1
        output2

  The only exception of this is when displaying raw output, i.e. with no preceding
  commands. Then one may use just the "::" directive to improve visibility E.g.

  .. code-block:: rst

     ::

        output1
        output2

* Any code blocks that show command line invocations must be preceded by ``$``, e.g.

  .. code-block:: rst

     .. code-block:: console

        $ ./configure --some-args
        $ make libs
        $ make ./ex1
        $ ./ex1 --some-args


* Environment variables such as ``$PETSC_DIR`` or ``$PATH`` must be preceded by
  ``$`` and be enclosed in double backticks, e.g.

  .. code-block:: rst

     Set ``$PETSC_DIR`` and ``$PETSC_ARCH``

* For internal links, use explicit labels, e.g

  .. code-block:: rst

     .. _sec_short_name:

     Section name
     ============

  and elsewhere (in any document),

  .. code-block:: rst

     See :ref:`link text <sec_short_name>`

* For internal links in the manual with targets outside the manual, always provide alt text
  so that the text will be  properly formatted in the :ref:`standalone PDF manual <sec_local_docs_latex>`, e.g.

   .. code-block:: rst

     PETSc has :doc:`mailing lists </community/mailing>`.

* We use the `sphinxcontrib-bibtex extension <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`__
  to include citations from BibTeX files.
  You must include ``.. bibliography::`` blocks at the bottom of a page including citations (`example <https://gitlab.com/petsc/petsc/-/raw/main/doc/manual/ksp.rst>`__).
  To cite the same reference in more than one page, use `this workaround <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#key-prefixing>`__ on one of them (`example <https://gitlab.com/petsc/petsc/-/raw/main/doc/developers/articles.rst>`__) [#bibtex_footnote]_.

* See special instructions on :any:`docs_images`.

* Prefer formatting styles that are easy to modify and maintain.  In particular, use of `list-table <https://docutils.sourceforge.io/docs/ref/rst/directives.html#list-table>`_ is recommended.

* When using external links with inline URLs, prefer to use `anonymous hyperlink references <https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#anonymous-hyperlinks>`__ with two trailing underscores, e.g.

  .. code-block:: rst

      `link text <https://external.org>`__

* To pluralize something with inline markup, e.g. ``DM``\s, escape the trailing character to avoid ``WARNING: Inline literal start-string without end-string``.

  .. code-block:: rst

      ``DM``\s

* Use restraint in adding new Sphinx extensions, in particular those which aren't
  widely-used and well-supported, or those with hidden system dependencies.

.. _docs_images:

Images
======

PETSc's documentation is tightly coupled to the source code and tests, and
is tracked in the primary PETSc Git repository. However, image files are
too large to directly track this way (especially because they persist in the integration branches' histories).

Therefore, we store image files in a separate git repository and clone it when
needed. Any new images required must be added to the currently-used branch of this repository.

Image Guidelines
----------------

* Whenever possible, use SVG files.  SVG is a web-friendly vector format and will be automatically converted to PDF using ``rsvg-convert`` [#svg_footnote]_
* Avoid large files and large numbers of images.
* Do not add movies or other non-image files.

Adding new images
-----------------

* Note the URL and currently-used branch (after ``-b``) for the upstream images repository, as used by the documentation build:

.. literalinclude:: /../doc/makefile
   :language: makefile
   :start-at: images:
   :lines: 2


* Decide where in ``doc/images`` a new image should go. Use the structure of the ``doc/`` tree itself as a guide.
* Create a Merge Request to the currently-used branch of the upstream images repository, adding this image [#maintainer_fast_image_footnote]_.
* Once this Merge Request is merged, you may make a :doc:`Merge Request to the primary PETSc repository </developers/integration>`, relying on the new image(s).

It may be helpful to place working copies of new image(s) in your local ``doc/images``
while iterating on documentation; just don't forget to update the upstream images repository.


Removing, renaming, moving or updating images
---------------------------------------------

Do not directly move, rename, or update images in the images repository.
Simply add a logically-numbered new version of the image.

If an image is not used in *any* :any:`integration branch <sec_integration_branches>` (``main`` or ``release``),
add it to the the top-level list of files to delete, in the images repository.

.. _docs_images_cleanup:

Cleaning up the images repository (maintainers only)
----------------------------------------------------

If the size of the image repository grows too large,

* Create a new branch ``main-X``, where ``X`` increments the current value
* Create a new commit deleting all files in the to-delete list and clearing the list
* Reset the new ``main-X`` to a single commit with this new, cleaned-up state
* Set ``main-X`` as the "default" branch on GitLab (or wherever it is hosted).
* Update both ``release`` and ``main`` in the primary PETSc repository to clone this new branch

.. _classic_docs_build:

Building Classic Documentation
==============================

Some of the documentation is built by a "classic" process as described below using the documentation tools listed below which are
automatically downloaded and installed if needed while building the PETSc documentation./

* `Sowing <http://ftp.mcs.anl.gov/pub/sowing/sowing.tar.gz>`__: a text processing tool developed by Bill Gropp.  This produces the PETSc manual pages; see the `Sowing documentation <http://wgropp.cs.illinois.edu/projects/software/sowing/doctext/doctext.htm>`__ and :ref:`manual_page_format`.
* `C2html <http://ftp.mcs.anl.gov/pub/petsc/c2html.tar.gz>`__: A text processing package. This generates the HTML versions of all the source code.

Sowing and C2html are build tools that do not use the compilers specified to PETSc's ``configure``, as they
need to work in cross-compilation environments. Thus, they default to using ``gcc``, ``g++``, and ``flex`` from
the user's environment (or ``configure`` options like ``--download-sowing-cxx``). Microsoft Windows users should install ``gcc``
etc. from Cygwin as these tools don't build with Microsoft or Intel Windows compilers.

.. rubric:: Footnotes

.. [#venv_footnote] This requires Python 3.3 or later, and you maybe need to install a package like ``python3-venv``.

.. [#bibtex_footnote] The extensions's `development branch <https://github.com/mcmtroffaes/sphinxcontrib-bibtex>`__ `supports our use case better <https://github.com/mcmtroffaes/sphinxcontrib-bibtex/pull/185>`__ (``:footcite:``), which can be investigated if a release is ever made.

.. [#svg_footnote] ``rsvg-convert`` is installable with your package manager, e.g., ``librsvg2-bin`` on Debian/Ubuntu systems).

.. [#maintainer_fast_image_footnote] Maintainers may directly push commits.
