# $Id: makefile,v 1.3 1998/03/19 02:05:06 balay Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = shvec.c
SOURCEF  =
SOURCEH  = 
OBJSC    = shvec.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/impls/shared/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

