# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $ 

ALL: lib

CFLAGS   = ${CPPFLAGS}  -D__SDIR__='"${LOCDIR}"' ${ADIC_INCLUDE}
SOURCEC  = adpetscadic.c petscadic.c
SOURCEF  =
SOURCEH  = 
OBJSC    = adpetscadic.o petscadic.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 1
LOCDIR   = src/adic/src/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

