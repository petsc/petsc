# $Id: makefile,v 1.2 1997/11/09 04:09:56 bsmith Exp bsmith $

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = aopart.c
SOURCEF  =
SOURCEH  = 
OBJSC    = aopart.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/ao/utils/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

