# $Id: makefile,v 1.26 1999/02/02 21:26:45 balay Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = mmbdiag.c mpibdiag.c
SOURCEF  =
SOURCEH  = mpibdiag.h
OBJSC    = mmbdiag.o mpibdiag.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = Mat
LOCDIR   = src/mat/impls/bdiag/mpi/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
