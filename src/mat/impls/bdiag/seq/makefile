# $Id: makefile,v 1.22 1998/04/23 23:45:06 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = bdiag.c bdiag2.c bdfact.c
SOURCEF  =
SOURCEH  = bdiag.h
OBJSC    = bdiag.o bdiag2.o bdfact.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/bdiag/seq/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
