
ALL: lib

CFLAGS   = ${PBGL_INCLUDE}
FFLAGS   =
SOURCEC	 = mpiaij.c mmaij.c mpiaijpc.c mpiov.c fdmpiaij.c mpiptap.c mpimatmatmult.c
SOURCEF	 =
SOURCEH	 = mpiaij.h
OBJSC	 = mpiaij.o mmaij.o mpiaijpc.o mpiov.o fdmpiaij.o mpiptap.o mpimatmatmult.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = spooles superlu_dist mumps csrperm crl pastix
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test
