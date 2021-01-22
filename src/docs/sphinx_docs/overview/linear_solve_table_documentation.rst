.. _doc_linsolve:

===================================================
Summary of Sparse Linear Solvers Available In PETSc
===================================================

Preconditioners
===============

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * -
     - Algorithm
     - Associated Type
     - Matrix Types
     - External Packages
     - Parallel
     - Complex
   * - Generic
     - Jacobi
     - ``PCJACOBI``
     - ``MATAIJ``, ``MATBAIJ``, ``MATSBAIJ``, ``MATDENSE``
     - ---
     - X
     - X
   * -
     - Point Block Jacobi
     - ``PCPBJACOBI``
     - ``MATAIJ``, ``MATBAIJ``, ``MATSBAIJ``, ``MATKAIJ``, ``MATMPISELL``, ``MATIS``
     - ---
     - X
     - X
   * -
     - Block Jacobi
     - ``PCBJACOBI``
     - ``MATAIJ``, ``MATBAIJ``, ``MATSBAIJ``
     - ---
     - X
     - X
   * -
     - SOR
     - ``PCSOR``
     - ``MATAIJ``, ``MATSEQDENSE``, ``MATSEQSBAIJ``
     - ---
     -
     - X
   * -
     - Point Block SOR
     -
     - ``MATSEQBAIJ`` (only for ``bs`` = 2,3,4,5)
     - ---
     -
     - X
   * -
     - Additive Schwarz
     - ``PCASM``
     - ``MATAIJ``, ``MATBAIJ``, ``MATSBAIJ``
     - ---
     - X
     - X
   * -
     - Deflation
     - ``PCDEFLATION``
     - All
     - ---
     - X
     - X
   * - Incomplete
     - ILU
     - ``PCILU``
     - ``MATSEQAIJ``, ``MATSEQBAIJ``
     - ---
     -
     - X
   * -
     - ILU with drop tolerance
     - ``PCILU``
     - ``MATSEQAIJ``
     - `SuperLU Sequential ILU solver
       <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MATSOLVERSUPERLU.html>`__
     -
     - X
   * -
     -
     - ``PCILU``
     - ``MATAIJ``
     - `Euclid/hypre
       <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCHYPRE.html>`__
     - X
     -
   * -
     - ICholesky
     - ``PCICC``
     - ``MATSEQAIJ``, ``MATSEQBAIJ``, ``MATSEQSBAIJ``
     - ---
     -
     - X
   * - Matrix Free
     - Infrastructure
     - ``PCSHELL``
     - All
     - ---
     - X
     - X
   * - Multigrid
     - Infrastructure
     - ``PCMG``
     - All
     - ---
     - X
     - X
   * -
     - Geometric
     -
     - All
     - ---
     - X
     - X
   * -
     - Smoothed Aggregation
     - ``PCGAMG``
     - ``MATAIJ``
     - ---
     - X
     - X
   * -
     - Structured Geometric
     - ``PCPFMG``
     - ``MATHYPRESTRUCT``
     - `hypre <https://hypre.readthedocs.io/en/latest/solvers-smg-pfmg.html>`__
     - X
     -
   * -
     - Classical Algebraic
     - ``PCHYPRE``, ``PCML``
     - ``MATAIJ``
     - `BoomerAMG/hypre
       <https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html>`__, `ML/Trilinos
       <https://trilinos.github.io/ml.html>`__
     - X
     -
   * -
     - Domain Decomposition
     - ``PCHPDDM``
     - ``MATAIJ``, ``MATBAIJ``, ``MATSBAIJ``, ``MATIS``
     - `HPDDM <https://github.com/hpddm/hpddm>`__
     - X
     - X
   * - Physics-based Splitting
     - Relaxation & Schur Complement
     - ``PCFIELDSPLIT``
     - ``MATAIJ``, ``MATBAIJ``, ``MATNEST``
     - ---
     - X
     - X
   * -
     - Least Squares Commutator
     - ``PCLSC``
     - ``MATSCHURCOMPLEMENT``
     - ---
     - X
     - X
   * - Approximate Inverse
     - AIV
     - ``PCHYPRE``, ``PCSPAI``
     - ``MATAIJ``
     - `Parasails/hypre <https://hypre.readthedocs.io/en/latest/solvers-parasails.html>`__, `SPAI <https://epubs.siam.org/doi/abs/10.1137/S1064827595294691?journalCode=sjoce3>`__
     - X
     -
   * - Substructuring
     - Balancing Neumann-Neumann
     - ``PCNN``
     - ``MATIS``
     - ---
     - X
     - X
   * -
     - Balancing Domain Decomposition
     - ``PCBDDC``
     - ``MATIS``
     - ---
     - X
     - X

-------------------------------

Direct Solvers
==============

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * -
     - Algorithm
     - Associated Type
     - Matrix Types
     - External Packages
     - Parallel
     - Complex
   * - Direct LU
     - LU
     - ``PCLU``
     - ``MATSEQAIJ``, ``MATSEQBAIJ``
     - ---
     -
     - X
   * -
     -
     -
     - ``MATSEQAIJ``
     - `MATLAB <https://www.mathworks.com/products/matlab.html>`__
     -
     - X
   * -
     -
     -
     - ``MATAIJ``
     - `PaStiX <http://pastix.gforge.inria.fr/files/README-txt.html>`__
     - X
     - X
   * -
     -
     -
     - ``MATAIJ``
     - `SuperLU <https://portal.nersc.gov/project/sparse/superlu/>`__
     - X
     - X
   * -
     -
     -
     - ``MATAIJ``, ``MATBAIJ``
     - `MUMPS <https://mumps.enseeiht.fr/>`__
     - X
     - X
   * -
     -
     -
     - ``MATSEQAIJ``
     - `ESSL <https://www.ibm.com/support/knowledgecenter/en/SSFHY8/essl_welcome.html>`__
     -
     -
   * -
     -
     -
     - ``MATSEQAIJ``
     - `UMPFPACK (SuiteSparse) <https://people.engr.tamu.edu/davis/suitesparse.html>`__
     -
     - X
   * -
     -
     -
     - ``MATSEQAIJ``
     - `KLU (SuiteSparse) <https://people.engr.tamu.edu/davis/suitesparse.html>`__
     -
     - X
   * -
     -
     -
     - ``MATSEQAIJ``
     - `LUSOL <https://web.stanford.edu/group/SOL/software/lusol/>`__
     -
     -
   * -
     -
     -
     - ``MATSEQAIJ``, ``MATSEQBAIJ``
     - `MKL Pardiso
       <https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top.html>`__
     -
     - X
   * -
     -
     -
     - ``MATMPIAIJ``, ``MATMPIBAIJ``
     - `MKL CPardiso
       <https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top.html>`__
     - X
     - X
   * -
     -
     -
     - ``MATDENSE``
     - `Elemental <https://github.com/elemental/Elemental>`__
     - X
     - X
   * - Direct Cholesky
     - Cholesky
     - ``PCCHOLESKY``
     - ``MATSEQAIJ``, ``MATSEQSBAIJ``
     - ---
     -
     - X
   * -
     -
     -
     - ``MATSBAIJ``
     - `PaStiX <http://pastix.gforge.inria.fr/files/README-txt.html>`__
     - X
     - X
   * -
     -
     -
     - ``MATSBAIJ``
     - `MUMPS <https://mumps.enseeiht.fr/>`__
     - X
     - X
   * -
     -
     -
     - ``MATSEQAIJ``, ``MATSEQSBAIJ``
     - `CHOLMOD (SuiteSparse) <https://people.engr.tamu.edu/davis/suitesparse.html>`__
     -
     - X
   * -
     -
     -
     - ``MATDENSE``
     - `Elemental <https://github.com/elemental/Elemental>`__
     - X
     - X
   * -
     -
     -
     - ``MATSEQSBAIJ``
     - `MKL Pardiso
       <https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top.html>`__
     -
     -
   * -
     -
     -
     - ``MATMPIAIJ``, ``MATMPIBAIJ``
     - `MKL CPardiso
       <https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top.html>`__
     - X
     -
   * - Direct QR
     - QR
     -
     - MATLAB
     - `MATLAB <https://www.mathworks.com/products/matlab.html>`__
     -
     -
   * -
     - XXt and XYt
     -
     - ``MATAIJ``
     - ---
     - X
     -

-------------------------------

Krylov Methods
==============

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * - Algorithm
     - Associated Type
     - External Packages
     - Parallel
     - Complex
   * - Richardson
     - ``KSPRICHARDSON``
     - ---
     - X
     - X
   * - Chebyshev
     - ``KSPCHEBYSHEV``
     - ---
     - X
     - X
   * - GMRES
     - ``KSPGMRES``
     - ---
     - X
     - X
   * - Flexible GMRES
     - ``KSPFGMRES``
     - ---
     - X
     - X
   * - LGMRES
     - ``KSPLGMRES``
     - ---
     - X
     - X
   * - Conjugate Gradient
     - ``KSPCG``
     - ---
     - X
     - X
   * - Conjugate Gradient Squared
     - ``KSPCGS``
     - ---
     - X
     - X
   * - Conjugate Gradient for Least Squares
     - ``KSPCGLS``
     - ---
     - X
     - X
   * - Conjugate Gradient on Normal Equations
     - ``KSPCGNE``
     - ---
     - X
     - X
   * - Bi-Conjugate Gradient
     - ``KSPBICG``
     - ---
     - X
     - X
   * - Stabilized Bi-Conjugate Gradient
     - ``KSPBCGS``
     - ---
     - X
     - X
   * - Transpose-free QMR
     - ``KSPTFQMR``
     - ---
     - X
     - X
   * - Conjugate Residual
     - ``KSPCR``
     - ---
     - X
     - X
   * - Generalized Conjugate Residual
     - ``KSPGCR``
     - ---
     - X
     - X
   * - Generalized Conjugate Residual (with inner normalization and deflated restarts)
     - ``KSPHPDDM``
     - `HPDDM <https://github.com/hpddm/hpddm>`__
     - X
     - X
   * - Minimum Residual
     - ``KSPMINRES``
     - ---
     - X
     - X
   * - LSQR
     - ``KSPLSQR``
     - ---
     - X
     - X
   * - SYMMLQ
     - ``KSPSYMMLQ``
     - ---
     - X
     - X
