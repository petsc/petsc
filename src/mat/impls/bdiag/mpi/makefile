# $Id: makefile,v 1.27 2000/08/17 04:51:34 bsmith Exp balay $ 

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
