# $Id: makefile,v 1.4 1998/04/23 23:30:22 balay Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = aopart.c
SOURCEF  =
SOURCEH  = 
OBJSC    = aopart.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/ao/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

