# $Id: makefile,v 1.29 2000/08/17 04:51:33 bsmith Exp balay $ 

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
