# $Id: makefile,v 1.3 1998/03/20 22:53:55 bsmith Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = aopart.c
SOURCEF  =
SOURCEH  = 
OBJSC    = aopart.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/ao/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

