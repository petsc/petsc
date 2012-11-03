
ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCEC	 = mpiaij.c mmaij.c mpiaijpc.c mpiov.c fdmpiaij.c mpiptap.c mpimatmatmult.c mpb_aij.c
SOURCEF	 =
SOURCEH	 = mpiaij.h
LIBBASE	 = libpetscmat
DIRS	 = superlu_dist mumps csrperm crl pastix mpicusp mpicusparse clique
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
