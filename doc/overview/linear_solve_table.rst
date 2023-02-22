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
     - Variable Point Block Jacobi
     - ``PCPBJACOBI``
     - ``MATAIJ``, ``MATBAIJ``, ``MATSBAIJ``
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
     - X
     - X
   * -
     - Point Block SOR
     -
     - ``MATSEQBAIJ`` (only for ``bs`` = 2,3,4,5)
     - ---
     - X
     - X
   * -
     - Kaczmarz
     - ``PCKACZMARZ``
     - ``MATAIJ``
     - ---
     - X
     - X
   * -
     - Additive Schwarz
     - ``PCASM``
     - ``MATAIJ``, ``MATBAIJ``, ``MATSBAIJ``
     - ---
     - X
     - X
   * -
     - Vanka/overlapping patches
     - ``PCPATCH``
     - ``MATAIJ``
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
       <../../manualpages/Mat/MATSOLVERSUPERLU.html>`__
     -
     - X
   * -
     -
     - ``PCILU``
     - ``MATAIJ``
     - Euclid/hypre (``PCHYPRE``)
     - X
     -
   * -
     - ICholesky
     - ``PCICC``
     - ``MATSEQAIJ``, ``MATSEQBAIJ``, ``MATSEQSBAIJ``
     - ---
     -
     - X
   * -
     - Algebraic recursive multilevel
     - ``PCPARMS``
     - ``MATSEQAIJ``
     - `pARMS <https://www-users.cse.umn.edu/~saad/software/pARMS/>`__
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
     - Smoothed Aggregation (ML)
     - ``PCML``
     - ``MATAIJ``
     - `ML/Trilinos <https://trilinos.github.io/ml.html>`__
     - X
     - X
   * -
     - Structured Geometric
     - ``PCPFMG``, ``PCSYSPFMG``, ``PCSMG``
     - ``MATHYPRESTRUCT``
     - `hypre <https://hypre.readthedocs.io/en/latest/solvers-smg-pfmg.html>`__
     - X
     -
   * -
     - Classical Algebraic
     - ``PCHYPRE``, ``PCAMGX``
     - ``MATAIJ``
     - `BoomerAMG/hypre
       <https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html>`__, `AmgX <https://developer.nvidia.com/amgx>`__
     - X
     -
   * -
     - Multi-group MG
     - ``PCHMG``
     - ``MATAIJ``
     - ---
     - X
     - X
   * -
     - Domain Decomposition
     - ``PCHPDDM``
     - ``MATAIJ``, ``MATBAIJ``, ``MATSBAIJ``, ``MATIS``
     - `HPDDM <https://github.com/hpddm/hpddm>`__
     - X
     - X
   * - Hierarchical matrices
     - :math:`\mathcal H^2`
     - ``PCH2OPUS``
     - ``MATHTOOL``, ``MATH2OPUS``
     - `H2OPUS <https://github.com/ecrc/h2opus>`__
     - X
     -
   * - Physics-based Splitting
     - Relaxation & Schur Complement
     - ``PCFIELDSPLIT``
     - ``MATAIJ``, ``MATBAIJ``, ``MATNEST``
     - ---
     - X
     - X
   * -
     - Galerkin composition
     - ``PCGALERKIN``
     - Any
     - ---
     - X
     - X
   * -
     - Additive/multiplicative
     - ``PCCOMPOSITE``
     - Any
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
   * - Parallel transformation
     - Redistribution
     - ``PCREDISTRIBUTE``
     - ``MATAIJ``
     - ---
     - X
     - X
   * -
     - Telescoping communicator
     - ``PCTELESCOPE``
     - ``MATAIJ``
     - ---
     - X
     - X
   * -
     - Distribute for MPI
     - ``PCMPI``
     - ``MATAIJ``
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
   * -
     - 2-level Schwarz wire basket
     - ``PCEXOTIC``
     - ``MATAIJ``
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
     - `MUMPS <https://mumps-solver.org/>`__
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
     - `MUMPS <https://mumps-solver.org/>`__
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
   * - Direct SVD
     - Singular value decomposition
     - ``PCSVD``
     - Any
     - ---
     - X
     - X
   * - Direct QR
     - QR
     - ``PCQR``
     - ``MATSEQAIJ``
     -  `SuiteSparse QR <https://people.engr.tamu.edu/davis/suitesparse.html>`__
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
   * - Deflated GMRES
     - ``KSPDGMRES``
     - ---
     - X
     -
   * - Two-stage with least squares residual minimization
     - ``KSPTSIRM``
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
   * - Nash Conjugate Gradient with trust region constraint
     - ``KSPNASH``
     - ---
     - X
     - X
   * - Conjugate Gradient with trust region constraint
     - ``KSPSTCG``
     - ---
     - X
     - X
   * - Gould et al Conjugate Gradient with trust region constraint
     - ``KSPGLTR``
     - ---
     - X
     - X
   * - Steinhaug Conjugate Gradient with trust region constraint
     - ``KSPQCG``
     - ---
     - X
     - X
   * - Left Conjugate Direction
     - ``KSPLCD``
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
   * - Improved Stabilized Bi-Conjugate Gradient
     - ``KSPIBCGS``
     - ---
     - X
     - X
   * - Transpose-free QMR
     - ``KSPTFQMR``
     - ---
     - X
     - X
   * - Tony Chan QMR
     - ``KSPTCQMR``
     - ---
     - X
     - X
   * - QMR BiCGStab
     - ``KSPQMRCGS``
     - ---
     - X
     - X
   * - Flexible Conjugate Gradients
     - ``KSPFCG``
     - ---
     - X
     - X
   * - Flexible stabilized Bi-Conjugate Gradients
     - ``KSPFBCGS``
     - ---
     - X
     - X
   * - Flexible stabilized Bi-Conjugate Gradients with fewer reductions
     - ``KSPFBCGSR``
     - ---
     - X
     - X
   * - Stabilized Bi-Conjugate Gradients with length :math:`\ell` recurrence
     - ``KSPBCGSL``
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
   * - FETI-DP (reduction to dual-primal sub-problem)
     - ``KSPFETIDP``
     - ---
     - X
     - X
   * - Gropp's overlapped reduction Conjugate Gradient
     - ``KSPGROPPCG``
     - ---
     - X
     - X
   * - Pipelined Conjugate Gradient
     - ``KSPPIPECG``
     - ---
     - X
     - X
   * - Pipelined Conjugate Gradient with residual replacement
     - ``KSPPIPECGRR``
     - ---
     - X
     - X
   * - Pipelined depth :math:`\ell` Conjugate Gradient
     - ``KSPPIPELCG``
     - ---
     - X
     - X
   * - Pipelined predict-and-recompute Conjugate Gradient
     - ``KSPPIPEPRCG``
     - ---
     - X
     - X
   * - Pipelined Conjugate Gradient over iteration pairs
     - ``KSPPIPECG2``
     - ---
     - X
     - X
   * - Pipelined flexible Conjugate Gradient
     - ``KSPPIPEFCG``
     - ---
     - X
     - X
   * - Pipelined stabilized Bi-Conjugate Gradients
     - ``KSPPIPEBCGS``
     - ---
     - X
     - X
   * - Pipelined Conjugate Residual
     - ``KSPPIPECR``
     - ---
     - X
     - X
   * - Pipelined flexible GMRES
     - ``KSPPIPEFGMRES``
     - ---
     - X
     - X
   * - Pipelined Generalized Conjugate Residual
     - ``KSPPIPEGCR``
     - ---
     - X
     - X
   * - Pipelined GMRES
     - ``KSPPGMRES``
     - ---
     - X
     - X
