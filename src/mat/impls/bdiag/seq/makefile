# $Id: makefile,v 1.21 1998/03/19 02:07:51 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = bdiag.c bdfact.c
SOURCEF  =
SOURCEH  = bdiag.h
OBJSC    = bdiag.o bdfact.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/bdiag/seq/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
