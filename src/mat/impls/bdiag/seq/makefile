# $Id: makefile,v 1.28 1999/02/02 21:26:41 balay Exp bsmith $ 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = bdiag.c bdiag2.c bdfact.c bdiag3.c
SOURCEF  =
SOURCEH  = bdiag.h
OBJSC    = bdiag.o bdiag2.o bdfact.o bdiag3.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = Mat
LOCDIR   = src/mat/impls/bdiag/seq/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
