
/**

 \page manual-user-page-acknowledgments Acknowledgments

We thank all %PETSc users for their many suggestions, bug reports, and
encouragement.  We especially thank David Keyes
for his valuable comments on the source code,
functionality, and documentation for %PETSc.



Some of the source code and utilities in %PETSc
have been written by
  - Asbjorn Hoiland Aarrestad - the explicit Runge-Kutta implementations;
  - Mark Adams - scalability features of MPIBAIJ matrices;
  - G. Anciaux and J. Roman - the interfaces to the partitioning packages PTScotch, Chaco, and Party;
  - Allison Baker - the flexible GMRES code and LGMRES;
  - Chad Carroll - Win32 graphics;
  - Ethan Coon - the PetscBag and many bug fixes;
  - Cameron Cooper - portions of the VecScatter routines;
  - Paulo Goldfeld - balancing Neumann-Neumann preconditioner;
  - Matt Hille;
  - Joel Malard - the BICGStab(l) implementation;
  - Paul Mullowney, enhancements to portions of the Nvidia GPU interface;
  - Dave May - the GCR implementation
  - Peter Mell - portions of the DA routines;
  - Richard Mills - the AIJPERM matrix format for the Cray X1 and universal F90 array interface;
  - Victor Minden - the NVidia GPU interface;
  - Todd Munson - the LUSOL (sparse solver in MINOS) interface and several Krylov methods;
  - Adam Powell - the PETSc Debian package,
  - Robert Scheichl - the MINRES implementation,
  - Kerry Stevens - the pthread based Vec and Mat classes plus the various thread pools
  - Karen Toonen - designed and implemented much of the PETSc web pages,
  - Desire Nuentsa Wakam - the deflated GMRES implementation,
  - Liyang Xu - the interface to PVODE (now Sundials/CVODE).

%PETSc uses routines from
  - BLAS;
  - LAPACK;
  - LINPACK - dense matrix factorization and solve; converted to C using `f2c` and then
                     hand-optimized for small matrix sizes, for block matrix data structures;
  - MINPACK - see page \pageref{sec_fdmatrix}, sequential matrix coloring routines for finite difference Jacobian
             evaluations; converted to C using `f2c`;
  - SPARSPAK - see page \pageref{sec_factorization}, matrix reordering routines, converted to C using `f2c`;
  - libtfs - the efficient, parallel direct solver developed by Henry Tufo and Paul Fischer for the direct solution of a coarse grid problem (a linear system with very few degrees of freedom per processor).


%PETSc interfaces to the following external software:
  - \link http://www.mcs.anl.gov/adifor ADIFOR \endlink -  automatic differentiation for the computation of sparse Jacobians,
  - \link http://www.cs.sandia.gov/CRF/chac.html Chaco \endlink -     A graph partitioning package, 
  - ESSL -         IBM's math library for fast sparse direct LU factorization,
  - Euclid  -   parallel ILU(k) developed by David Hysom, accessed through the Hypre interface,
  - \link http://www.llnl.gov/CASC/hypre Hypre \endlink -    the LLNL preconditioner library, 
  - \link http://www.sbsi-sol-optimize.com/ LUSOL \endlink -       sparse LU factorization code (part of MINOS) developed by Michael Saunders,
                      Systems Optimization Laboratory, Stanford University,
  - Mathematica -  see page \pageref{ch_mathematica},
  - MATLAB -      see page \pageref{ch_matlab},
  - \link http://www.enseeiht.fr/lima/apo/MUMPS/credits.html MUMPS \endlink -      see page \pageref{sec_externalsol}, MUltifrontal Massively Parallel sparse direct Solver developed by Patrick Amestoy, Iain Duff, Jacko Koster, and Jean-Yves L'Excellent,
  - \link http://www-users.cs.umn.edu/~karypis/metis/ ParMeTiS \endlink -     see page \pageref{sec_partitioning}, parallel graph partitioner,
  - \link http://www.uni-paderborn.de/fachbereich/AG/monien/RESEARCH/PART/party.html  Party \endlink  -     A graph partitioning package,
  - PaStiX -     Parallel LU and Cholesky solvers,
  - \link http://www.labri.fr/Perso/~pelegrin/scotch/ PTScotch \endlink -    A graph partitioning package, 
  - \link http://www.sam.math.ethz.ch/~grote/spai/ SPAI \endlink -        for parallel sparse approximate inverse preconditiong,
  - \link http://www.llnl.gov/CASC/sundials/ Sundial/CVODE \endlink - see page \pageref{sec_sundials}, parallel ODE integrator,
  - \link http://www.nersc.gov/~xiaoye/SuperLU SuperLU and SuperLU_Dist \endlink - see page \pageref{sec_externalsol},
                    the efficient sparse LU codes developed by Jim Demmel,  Xiaoye S. Li, and John Gilbert,
  - \link http://software.sandia.gov/trilinos/ Trilinos/ML \endlink - Sandia's main multigrid preconditioning package, ,
  - \link http://www.cise.ufl.edu/research/sparse/umfpack/ UMFPACK \endlink - see page \pageref{sec_externalsol},
                    developed by Timothy A. Davis.
These are all optional packages and do not need to be installed to use PETSc.

%PETSc software is developed and maintained with
  - \link http://git-scm.com/ Git \endlink revision control system
  - Emacs editor


%PETSc documentation has been generated using
  - the \link http://www.cs.uiuc.edu/~wgropp/projects/software/sowing/index.htm text processing tools \endlink developed by Bill Gropp
  - c2html
  - pdflatex
  - python

*/

