
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC	 = mpiaij.c mmaij.c mpiaijpc.c mpiov.c fdmpiaij.c mpiptap.c
SOURCEF	 =
SOURCEH	 = mpiaij.h
OBJSC	 = mpiaij.o mmaij.o mpiaijpc.o mpiov.o fdmpiaij.o mpiptap.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = spooles superlu_dist mumps
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
