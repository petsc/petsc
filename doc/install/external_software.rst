.. _doc_externalsoftware:

***************************
Supported External Software
***************************

PETSc interfaces to the following optional external software. See :ref:`installing
packages <doc_config_externalpack>` for more information on downloading and installing
these softwares, as well as the :doc:`linear solve table
</overview/linear_solve_table>` for more
information on the intended use-cases for each software.

List Of Software
================

.. glossary::
   :sorted:

   `AMD <http://www.cise.ufl.edu/research/sparse/amd/>`__
      Approximate minimum degree orderings.

   `BLAS/LAPACK <https://www.netlib.org/lapack/lug/node11.html>`__
      Optimizes linear algebra kernels

   `CUDA <https://developer.nvidia.com/cuda-toolkit>`__
      A parallel computing platform and application programming interface model created by
      Nvidia.

   `Chaco <http://www.cs.sandia.gov/CRF/chac.html>`__
      A graph partitioning package.

   `ESSL <https://www.ibm.com/support/knowledgecenter/en/SSFHY8/essl_welcome.html>`__
      IBM's math library for fast sparse direct LU factorization.

   `FFTW <http://www.fftw.org/>`__
      Fastest Fourier Transform in the West, developed at MIT by Matteo Frigo and
      Steven G. Johnson.

   `Git <https://git-scm.com/>`__
      Distributed version control system

   `HDF5 <http://portal.hdfgroup.org/display/support>`__
      A data model, library, and file format for storing and managing data.

   `Hypre <https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods>`__
      LLNL preconditioner library.

   `Kokkos <https://github.com/kokkos/kokkos>`__
      A programming model in C++ for writing performance portable applications targeting
      all major HPC platforms.

   `LUSOL <https://web.stanford.edu/group/SOL/software/lusol/>`__
      Sparse LU factorization and solve portion of MINOS, Michael Saunders, Systems
      Optimization Laboratory, Stanford University.

   `Mathematica <http://www.wolfram.com/>`__
      A general multi-paradigm computational language developed by Wolfram Research.

   `MATLAB <https://www.mathworks.com/>`__
      A proprietary multi-paradigm programming language and numerical computing
      environment developed by MathWorks.

   `MUMPS <http://mumps.enseeiht.fr/>`__
      MUltifrontal Massively Parallel sparse direct Solver.

   `MeTis/ParMeTiS <https://www-users.cs.umn.edu/~karypis/metis/>`__
      A serial/parallel graph partitioner.

   `Party <https://www.researchgate.net/publication/2736581_PARTY_-_A_software_library_for_graph_partitioning>`__
      A graph partitioning package.

   `PaStiX <https://gforge.inria.fr/projects/pastix/>`__
      A parallel LU and Cholesky solver package.

   `PTScotch <http://www.labri.fr/perso/pelegrin/scotch/>`__
      A graph partitioning package.

   `SPAI <https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-09766-4_144>`__
      Parallel sparse approximate inverse preconditioning.

   `SPRNG <http://www.sprng.org/>`__
      The Scalable Parallel Random Number Generators Library.

   `Sundial/CVODE <https://computation.llnl.gov/projects/sundials>`__
      The LLNL SUite of Nonlinear and DIfferential/ALgebraic equation Solvers.

   `SuperLU <https://crd-legacy.lbl.gov/~xiaoye/SuperLU/#superlu>`__ and `SuperLU_Dist <https://crd-legacy.lbl.gov/~xiaoye/SuperLU/#superlu_dist>`__
      Robust and efficient sequential and parallel direct sparse solves.

   `Trilinos/ML <http://trilinos.org/>`__
      Multilevel Preconditioning Package. Sandia's main multigrid preconditioning package.

   `SuiteSparse, including KLU, UMFPACK, and CHOLMOD <http://faculty.cse.tamu.edu/davis/suitesparse.html>`__
      Sparse direct solvers, developed by Timothy A. Davis.

   `ViennaCL <http://viennacl.sourceforge.net/>`__
      Linear algebra library providing matrix and vector operations using OpenMP, CUDA,
      and OpenCL.


Additional Software
===================

PETSc contains modifications of routines from:

- LINPACK (matrix factorization and solve; converted to C using f2c and then
  hand-optimized for small matrix sizes)
- MINPACK (sequential matrix coloring routines for finite difference Jacobian evaluations;
  converted to C using f2c)
- SPARSPAK (matrix reordering routines, converted to C using f2c, this is the PUBLIC
  DOMAIN version of SPARSPAK)
- libtfs (the scalable parallel direct solver created and written by Henry Tufo and Paul
  Fischer).

Instrumentation of PETSc:

- PETSc can be instrumented using the `TAU
  <http://www.cs.uoregon.edu/research/paracomp/tau/tautools/>`__ package (check
  :ref:`installation <doc_config_tau>` instructions).

PETSc documentation has been generated using:

- `Sowing <http://wgropp.cs.illinois.edu/projects/software/sowing/index.html>`__
- `c2html <http://user.cs.tu-berlin.de/~schintke/x2html/index.html>`__
- `Python <https://www.python.org/>`__
- `Sphinx <https://www.sphinx-doc.org/en/master/>`__
