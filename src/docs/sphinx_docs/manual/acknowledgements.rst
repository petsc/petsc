Acknowledgments
---------------

We thank all PETSc users for their many suggestions, bug reports, and
encouragement.

Recent contributors to PETSc can be seen by visualizing the history of
the PETSc git repository, for example at
`github.com/petsc/petsc/graphs/contributors <https://github.com/petsc/petsc/graphs/contributors>`__.

Earlier contributors to PETSc include:

-  Asbjorn Hoiland Aarrestad - the explicit Runge-Kutta implementations
   (````\ ```TSRK`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/TS/TSRK.html#TSRK>`__\ ````);

-  G. Anciaux and J. Roman - the interfaces to the partitioning packages
   PTScotch, Chaco, and Party;

-  Allison Baker - the flexible GMRES
   (````\ ```KSPFGMRES`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/KSP/KSPFGMRES.html#KSPFGMRES>`__\ ````)
   and LGMRES
   (````\ ```KSPLGMRES`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/KSP/KSPLGMRES.html#KSPLGMRES>`__\ ````)
   code;

-  Chad Carroll - Win32 graphics;

-  Ethan Coon - the
   ````\ ```PetscBag`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Sys/PetscBag.html#PetscBag>`__\ ````
   and many bug fixes;

-  Cameron Cooper - portions of the
   ````\ ```VecScatter`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Vec/VecScatter.html#VecScatter>`__\ ````
   routines;

-  Patrick Farrell -
   ````\ ```PCPATCH`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/PC/PCPATCH.html#PCPATCH>`__\ ````
   and
   ````\ ```SNESPATCH`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/SNES/SNESPATCH.html#SNESPATCH>`__\ ````;

-  Paulo Goldfeld - the balancing Neumann-Neumann preconditioner
   (````\ ```PCNN`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/PC/PCNN.html#PCNN>`__\ ````);

-  Matt Hille;

-  Joel Malard - the BICGStab(l) implementation
   (````\ ```KSPBCGSL`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/KSP/KSPBCGSL.html#KSPBCGSL>`__\ ````);

-  Paul Mullowney, enhancements to portions of the Nvidia GPU interface;

-  Dave May - the GCR implementation
   (````\ ```KSPGCR`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/KSP/KSPGCR.html#KSPGCR>`__\ ````);

-  Peter Mell - portions of the
   ````\ ```DMDA`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/DMDA/DMDA.html#DMDA>`__\ ````
   routines;

-  Richard Mills - the ``AIJPERM`` matrix format
   (````\ ```MATAIJPERM`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Mat/MATAIJPERM.html#MATAIJPERM>`__\ ````)
   for the Cray X1 and universal F90 array interface;

-  Victor Minden - the NVIDIA GPU interface;

-  Lawrence Mitchell -
   ````\ ```PCPATCH`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/PC/PCPATCH.html#PCPATCH>`__\ ````
   and
   ````\ ```SNESPATCH`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/SNES/SNESPATCH.html#SNESPATCH>`__\ ````;

-  Todd Munson - the LUSOL (sparse solver in MINOS) interface
   (````\ ```MATSOLVERLUSOL`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Mat/MATSOLVERLUSOL.html#MATSOLVERLUSOL>`__\ ````)
   and several Krylov methods;

-  Adam Powell - the PETSc Debian package;

-  Robert Scheichl - the MINRES implementation
   (````\ ```KSPMINRES`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/KSP/KSPMINRES.html#KSPMINRES>`__\ ````);

-  Kerry Stevens - the pthread-based
   ````\ ```Vec`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Vec/Vec.html#Vec>`__\ ````
   and
   ````\ ```Mat`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Mat/Mat.html#Mat>`__\ ````
   classes plus the various thread pools (no longer available);

-  Karen Toonen - design and implementation of much of the PETSc web
   pages;

-  Desire Nuentsa Wakam - the deflated GMRES implementation
   (````\ ```KSPDGMRES`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/KSP/KSPDGMRES.html#KSPDGMRES>`__\ ````);

-  Florian Wechsung -
   ````\ ```PCPATCH`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/PC/PCPATCH.html#PCPATCH>`__\ ````
   and
   ````\ ```SNESPATCH`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/SNES/SNESPATCH.html#SNESPATCH>`__\ ````;

-  Liyang Xu - the interface to PVODE, now SUNDIALS/CVODE
   (````\ ```TSSUNDIALS`` <https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/TS/TSSUNDIALS.html#TSSUNDIALS>`__\ ````);

PETSc source code contains modified routines from the following public
domain software packages:

-  LINPACK - dense matrix factorization and solve; converted to C using
   ``f2c`` and then hand-optimized for small matrix sizes, for block
   matrix data structures;

-  MINPACK - see page ; sequential matrix coloring routines for finite
   difference Jacobian evaluations; converted to C using ``f2c``;

-  SPARSPAK - see page ; matrix reordering routines, converted to C
   using ``f2c``;

-  libtfs - the efficient, parallel direct solver developed by Henry
   Tufo and Paul Fischer for the direct solution of a coarse grid
   problem (a linear system with very few degrees of freedom per
   processor).

PETSc interfaces to the following external software:

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

-  MATLAB - see page ;

-  | Metis/ParMeTiS - see page , parallel graph partitioner,
   | https://www-users.cs.umn.edu/~karypis/metis/

-  | MUMPS - see page , MUltifrontal Massively Parallel sparse direct
     Solver developed by Patrick Amestoy, Iain Duff, Jacko Koster, and
     Jean-Yves L’Excellent;
   | http://www.enseeiht.fr/lima/apo/MUMPS/credits.html

-  | Party - A graph partitioning package;
   | http://www2.cs.uni-paderborn.de/cs/ag-monien/PERSONAL/ROBSY/party.html

-  | PaStiX - Parallel sparse LU and Cholesky solvers;
   | http://pastix.gforge.inria.fr/

-  | PTScotch - A graph partitioning package;
   | http://www.labri.fr/Perso/~pelegrin/scotch/

-  | SPAI - for parallel sparse approximate inverse preconditioning;
   | https://cccs.unibas.ch/lehre/software-packages/

-  | SuiteSparse - sequential sparse solvers, see page , developed by
     Timothy A. Davis;
   | http://faculty.cse.tamu.edu/davis/suitesparse.html

-  | SUNDIALS/CVODE - see page , parallel ODE integrator;
   | https://computation.llnl.gov/projects/sundials

-  | SuperLU and SuperLU_Dist - see page , the efficient sparse LU codes
     developed by Jim Demmel, Xiaoye S. Li, and John Gilbert;
   | https://crd-legacy.lbl.gov/~xiaoye/SuperLU

-  | STRUMPACK - the STRUctured Matrix Package;
   | https://portal.nersc.gov/project/sparse/strumpack/

-  | Triangle and Tetgen - mesh generation packages;
   | https://www.cs.cmu.edu/~quake/triangle.html
   | http://wias-berlin.de/software/tetgen/

-  | Trilinos/ML - Sandia’s main multigrid preconditioning package;
   | https://software.sandia.gov//trilinos/,

-  | Zoltan - graph partitioners from Sandia National Laboratory;
   | http://www.cs.sandia.gov/zoltan/

These are all optional packages and do not need to be installed to use
PETSc.

PETSc software is developed and maintained using

Emacs editor

`Git <https://git-scm.com/>`__ revision control system

Python

PETSc documentation has been generated using

| Sowing text processing tools developed by Bill Gropp
| http://wgropp.cs.illinois.edu/projects/software/sowing/

c2html

pdflatex

python

 

[ch_index]

[sec:bib]

|image|
