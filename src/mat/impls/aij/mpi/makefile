# $Id: makefile,v 1.39 1999/02/02 21:26:28 balay Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC	 = mpiaij.c mmaij.c mpiaijpc.c mpiov.c fdmpiaij.c
SOURCEF	 =
SOURCEH	 = mpiaij.h
OBJSC	 = mpiaij.o mmaij.o mpiaijpc.o mpiov.o fdmpiaij.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
