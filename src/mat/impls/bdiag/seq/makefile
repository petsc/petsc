# $Id: makefile,v 1.20 1997/09/10 18:48:19 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = bdiag.c bdfact.c
SOURCEF  =
SOURCEH  = bdiag.h
OBJSC    = bdiag.o bdfact.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/bdiag/seq/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
