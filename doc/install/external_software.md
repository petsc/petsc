(doc_externalsoftware)=

# Supported External Software

PETSc interfaces with many optional external software packages. See {ref}`installing
packages <doc_config_externalpack>` for more information on downloading and installing
these software, as well as the {doc}`linear solver table
</overview/linear_solve_table>` for more
information on the intended use-cases for each software.

## Partial List Of Software

- [AMD](https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/AMD) Approximate minimum degree orderings.
- [BLAS/LAPACK](https://www.netlib.org/lapack/lug/node11.html) Optimizes linear algebra kernels (always available).
- [CUDA](https://developer.nvidia.com/cuda-toolkit) A parallel computing platform and application programming interface model created by NVIDIA.
- [Chaco](https://bitbucket.org/petsc/pkg-chaco) A graph partitioning package developed by Bruce Hendrickson and Robert Leland at Sandia National Laboratories and maintained by PETSc.
- [ESSL](https://www.ibm.com/support/knowledgecenter/en/SSFHY8/essl_welcome.html) IBM's math library for fast sparse direct LU factorization.
- [FFTW](https://www.fftw.org/) Fastest Fourier Transform in the West, developed at MIT by Matteo Frigo and Steven G. Johnson.
- [Git](https://git-scm.com/) Distributed version control system
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) A data model, library, and file format for storing and managing data.
- [Hypre](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods) LLNL preconditioner library.
- [Kokkos](https://github.com/kokkos/kokkos) A programming model in C++ for writing performance portable applications targeting all major HPC platforms
- [LUSOL](https://web.stanford.edu/group/SOL/software/lusol/) Sparse LU factorization and solve portion of MINOS, Michael Saunders, Systems Optimization Laboratory, Stanford University.
- [Mathematica](https://www.wolfram.com/) A general multi-paradigm computational language developed by Wolfram Research.
- [MATLAB](https://www.mathworks.com/) A proprietary multi-paradigm programming language and numerical computing environment developed by MathWorks.
- [MUMPS](https://mumps-solver.org/) MUltifrontal Massively Parallel sparse direct Solver.
- [MeTis](https://github.com/KarypisLab/METIS) and [ParMeTiS](https://github.com/KarypisLab/PARMETIS) serial/parallel graph partitioners.
- [Party](https://www.researchgate.net/publication/2736581_PARTY_-_A_software_library_for_graph_partitioning) A graph partitioning package.
- [PaStiX](https://solverstack.gitlabpages.inria.fr/pastix/) A parallel LU and Cholesky solver package.
- [PFLARE](https://github.com/PFLAREProject/PFLARE) Reduction-based algebraic multigrid (AIR) and approximate-inverse preconditioners for nonsymmetric systems (`PCAIR`, `PCPFLAREINV`).
- [PTScotch](https://www.labri.fr/perso/pelegrin/scotch/) A graph partitioning package.
- [SPAI](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-09766-4_144) Parallel sparse approximate inverse preconditioning.
- [SPRNG](https://www.cs.fsu.edu/~sprng/) The Scalable Parallel Random Number Generators Library.
- [SUNDIALS/CVODE](https://computing.llnl.gov/projects/sundials) The LLNL SUite of Nonlinear and DIfferential/ALgebraic equation Solvers.
- [SuperLU](https://github.com/xiaoyeli/superlu) and [SuperLU_DIST](https://github.com/xiaoyeli/superlu_dist) Robust and efficient sequential and parallel direct sparse solves.
- [Trilinos/ML](https://trilinos.github.io/) Multilevel Preconditioning Package.
- [SuiteSparse, including KLU, UMFPACK, and CHOLMOD](https://github.com/DrTimothyAldenDavis/SuiteSparse) Sparse direct solvers, developed by Timothy A. Davis.
- [ViennaCL](https://github.com/viennacl/viennacl-dev) An unmaintained linear algebra library that provides matrix and vector operations using OpenMP, CUDA, and OpenCL.

## Additional Software

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

- PETSc can be instrumented using the [TAU](https://www.cs.uoregon.edu/research/tau/home.php) package (check
  {ref}`installation <doc_config_tau>` instructions).

PETSc documentation has been generated using:

- [Sowing](https://bitbucket.org/petsc/pkg-sowing)
- [c2html](https://sources.debian.org/copyright/license/c2html/)
- [Python](https://www.python.org/)
- [Sphinx](https://www.sphinx-doc.org/en/master/)
