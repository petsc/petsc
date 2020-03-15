========================
PETSc Developer's Manual
========================

| Prepared by
| S. Kruger\ :sup:`1`, P. Sanan\ :sup:`2`, and B. Smith\ :sup:`3`
| :sup:`1`\ Tech-X Corporation
| :sup:`2`\ Institute of Geophysics, ETH Zurich
| :sup:`3`\ Mathematics and Computer Science Division, Argonne National Laboratory

| This material was based upon work supported by the Office of Science, Office of
| Advanced Scientific Computing Research, U.S. Department of Energy, under Contract
| DE-AC02-06CH11357.

PETSc is an extensible software library for scientific computation. This
document provides information for PETSc developers and those wishing to
contribute to PETSc. The text assumes that you are familiar with PETSc
and have access to PETSc source code and documentation (see
https://gitlab.com/petsc/petsc and https://www.mcs.anl.gov/petsc) including the PETSc :doc:`../manual/index` :cite:`petsc-user-ref` .
Higher-level views of PETSc can be found in :cite:`s2011`,
:cite:`bgms00`, :cite:`miss-paper`, :cite:`bgms98`, and :cite:`petsc-efficient`.

Before contributing code to PETSc, please read the :doc:`style`.
Information on how to submit patches and pull requests to PETSc can be
found at https://www.mcs.anl.gov/petsc/developers/index.html.

Please direct all comments and questions regarding PETSc design and
development to petsc-dev@mcs.anl.gov. Note that all *bug reports and
questions regarding the use of PETSc* should be directed to
petsc-maint@mcs.anl.gov.

.. toctree::
   :maxdepth: 2

   responding
   style
   kernel
   objects
   callbacks
   matrices
   testing

References
~~~~~~~~~~

.. bibliography:: ../../tex/petsc.bib
   :filter: docname in docnames

.. bibliography:: ../../tex/petscapp.bib
   :filter: docname in docnames
