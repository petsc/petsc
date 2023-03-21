
.. _acknowledgements:

================
Acknowledgements
================

We thank all PETSc/Tao/petsc4py users for their many suggestions, bug reports, and encouragement.

Recent contributors to PETSc are listed in the `repository system
<https://gitlab.com/petsc/petsc>`__. The history can be visualized at
`github.com/petsc/petsc/graphs/contributors <https://github.com/petsc/petsc/graphs/contributors>`__.

Earlier contributors to PETSc that are not captured in the repository system include:

* Asbjorn Hoiland Aarrestad, the explicit Runge-Kutta implementations, ``TSRK``.

* Guillaume Anciaux and Jose E. Roman, the interfaces to the partitioning packages PTScotch, Chaco, and Party.

* Allison Baker, ``KSPFGMRES`` and ``KSPLGMRES``.

* Chad Carroll, the Win32 graphics.

* Ethan Coon, `PetscBag` and many bug fixes.

* Cameron Cooper, portions of ``VecScatter`` routines.

* Patrick Farrell and Florian Wechsung, ``PCPATCH`` and ``SNESPATCH``.

* Paulo Goldfeld, early versions of the balancing Neumann-Neumann preconditioner ``PCNN``.

* Matt Hille.

* Joel Malard, ``KSPBCGS``.

* Paul Mullowney, enhancements to portions of the original CUDA GPU interface.

* Dave May, ``KSPGCR``.

* Peter Mell, portions of ``DMDA`` routines.

* Richard Mills, the ``MATAIJPERM`` matrix format for the Cray X1; universal F90 array
  interface; enhancements to ``KSPIBCGS``; the ``MATAIJMKL`` matrix subclass.

* Victor Minden, the original CUDA GPU interface.

* Todd Munson, ``MATSOLVERLUSOL`` as well as ``KSPNASH``, ``KSPSTCG``, and
  ``KSPGLTR``.

* Robert Scheichl, the original ``KSPMINRES`` implementation.

* Karen Toonen, designed and implemented most of the original PETSc web pages.

* Desire Nuentsa Wakam, ``KSPDGMRES``.

* Liyang Xu, the interface to PVODE (now Sundials/CVODE) ``TSSUNDIALS``.

The Toolkit for Advanced Optimization (Tao) developers especially thank Jorge Moré
for his leadership, vision, and effort on previous versions of Tao.  Tao has
also benefited from the work of various researchers who have provided solvers, test problems,
and interfaces. In particular, we acknowledge: Adam Denchfield, Elizabeth Dolan, Evan Gawlik,
Michael Gertz, Xiang Huang, Lisa Grignon, Manojkumar Krishnan, Gabriel Lopez-Calva,
Jarek Nieplocha, Boyana Norris, Hansol Suh, Stefan Wild, Limin Zhang, and
Yurii Zinchenko.

PETSc source code contains modified routines from the following public
domain software packages:

-  LINPACK - dense matrix factorization and solve; converted to C using
   ``f2c`` and then hand-optimized for small matrix sizes, for block
   matrix data structures;

-  MINPACK - sequential matrix coloring routines for finite
   difference Jacobian evaluations; converted to C using ``f2c``;

-  SPARSPAK -  matrix reordering routines, converted to C
   using ``f2c``;

-  libtfs - the efficient, parallel direct solver developed by Henry
   Tufo and Paul Fischer for the direct solution of a coarse grid
   problem (a linear system with very few degrees of freedom per
   processor).

PETSc interfaces to many external software packages including:

-  BLAS and LAPACK - numerical linear algebra;

-  | Chaco - A graph partitioning package;
   | http://www.cs.sandia.gov/CRF/chac.html

-  | Elemental - Jack Poulson’s parallel dense matrix solver package;
   | http://libelemental.org/

-  | HDF5 - the data model, library, and file format for storing and
     managing data,
   | https://support.hdfgroup.org/HDF5/

-  | hypre - the LLNL preconditioner library;
   | https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods

-  | LUSOL - sparse LU factorization code (part of MINOS) developed by
     Michael Saunders, Systems Optimization Laboratory, Stanford
     University;
   | http://www.sbsi-sol-optimize.com/

-  MATLAB

-  | Metis/ParMeTiS -  parallel graph partitioner,
   | https://www-users.cs.umn.edu/~karypis/metis/

-  | MUMPS -  MUltifrontal Massively Parallel sparse direct
     Solver developed by Patrick Amestoy, Iain Duff, Jacko Koster, and
     Jean-Yves L’Excellent;
   | https://mumps-solver.org/

-  | Party - A graph partitioning package;

-  | PaStiX - Parallel sparse LU and Cholesky solvers;
   | http://pastix.gforge.inria.fr/

-  | PTScotch - A graph partitioning package;
   | http://www.labri.fr/Perso/~pelegrin/scotch/

-  | SPAI - for parallel sparse approximate inverse preconditioning;
   | https://cccs.unibas.ch/lehre/software-packages/

-  | SuiteSparse - sequential sparse solvers, developed by
     Timothy A. Davis;
   | http://faculty.cse.tamu.edu/davis/suitesparse.html

-  | SUNDIALS/CVODE - (now an out-dated version), parallel ODE integrator;
   | https://computation.llnl.gov/projects/sundials

-  | SuperLU and SuperLU_Dist - the efficient sparse LU codes
     developed by Jim Demmel, Xiaoye S. Li, and John Gilbert;
   | https://crd-legacy.lbl.gov/~xiaoye/SuperLU

-  | STRUMPACK - the STRUctured Matrix Package;
   | https://portal.nersc.gov/project/sparse/strumpack/

-  | Triangle and Tetgen - mesh generation packages;
   | https://www.cs.cmu.edu/~quake/triangle.html
   | http://wias-berlin.de/software/tetgen/

-  | Trilinos/ML - Sandia’s main multigrid preconditioning package;
   | https://trilinos.github.io/

-  | Zoltan - graph partitioners from Sandia National Laboratory;
   | http://www.cs.sandia.gov/zoltan/

These are all optional packages and do not need to be installed to use
PETSc.

PETSc software is developed and maintained using

* `Git <https://git-scm.com/>`__ revision control system

PETSc documentation has been generated using

* https://www.sphinx-doc.org
* `Sowing text processing tools developed by Bill Gropp <http://wgropp.cs.illinois.edu/projects/software/sowing/>`__
* c2html
