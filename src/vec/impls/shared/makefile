# $Id: makefile,v 1.2 1997/11/28 16:18:25 bsmith Exp balay $ 

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = shvec.c
SOURCEF  =
SOURCEH  = 
OBJSC    = shvec.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/impls/shared/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

