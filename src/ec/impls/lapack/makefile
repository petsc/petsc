# $Id: makefile,v 1.179 1997/08/15 16:24:04 bsmith Exp bsmith $

ALL: lib


CFLAGS   = ${CPPFLAGS}  -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = elapack.c
SOURCEH  = 
SOURCEF  = 
OBJSC    = elapack.o
OBJSF    = 
LIBBASE  = libpetscec
MANSEC   = 5
LOCDIR   = src/ec/impls/lapack/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
