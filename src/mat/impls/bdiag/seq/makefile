# $Id: makefile,v 1.30 2001/08/22 18:00:36 balay Exp $ 

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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
