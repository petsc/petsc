==============================
Developing PETSc Documentation
==============================

.. toctree::
   :maxdepth: 2

.. _docs_build:

Building Documentation
======================

The documentation tools listed below (except for pdflatex) are
automatically downloaded and installed by ``./configure``.

* `Sowing <http://ftp.mcs.anl.gov/pub/sowing/sowing.tar.gz>`__: a text processing tool developed by Bill Gropp.  This produces the PETSc manual pages; see the `Sowing documentation <http://wgropp.cs.illinois.edu/projects/software/sowing/doctext/doctext.htm>`__ and :ref:`manual_page_format`.
* `C2html <http://ftp.mcs.anl.gov/pub/petsc/c2html.tar.gz>`__: A text processing package. This generates the HTML versions of all the source code.
* A version of pdflatex, for example in  `Tex Live <http://www.tug.org/texlive/>`__.  This package might already be installed on most systems. It is required to generate the users manual (part of the PETSc documentation).

Note: Sowing and c2html have additional dependencies like gcc, g++, and flex and do not
use compilers specified to PETSc configure. [Windows users please install the corresponding
cygwin packages]

Once pdflatex is in your ``PATH``, you can build the documentation with:

.. code-block:: bash

    make alldoc LOC=${PETSC_DIR}

(Note that this does not include :ref:`sphinx_documentation`).

To get a quick preview of manual pages from a single source directory (mainly to debug the manual page syntax):

.. code-block:: bash

    cd $PETSC_DIR/src/snes/interface
    make LOC=$PETSC_DIR manualpages_buildcite
    browse $PETSC_DIR/docs/manualpages/SNES/SNESCreate.html  # or suitable command to open the HTML page in a browser

.. _sphinx_documentation:

Sphinx Documentation
====================

The Sphinx documentation is currently not integrated into the main docs build as described
in :ref:`docs_build`.

`ReadTheDocs <readthedocs.org>`__ generates the Sphinx-based documentation found at
https://docs.petsc.org from the `PETSc git repository <https://gitlab.com/petsc/petsc>`__.

See ``README.md`` in the ``sphinx_docs`` directory for information on building locally.

Sphinx Documentation Guidelines
-------------------------------

* Use the ``includeliteral`` directive to directly include pieces of source code, as in
  the following example. Note that an "absolute" path has been used, which means
  relative to the root for the Sphinx docs (where ``conf.py`` is found).

.. code-block:: rst

    .. literalinclude:: /../../../src/sys/error/err.c
       :language: c
       :start-at: PetscErrorCode PetscError(
       :end-at: PetscFunctionReturn(0)
       :append: }
