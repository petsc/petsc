# $Id: makefile,v 1.18 1999/02/02 21:26:57 balay Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC	 = mpibaij.c mmbaij.c baijov.c
SOURCEF	 =
SOURCEH	 = mpibaij.h
OBJSC	 = mpibaij.o mmbaij.o baijov.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/baij/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
