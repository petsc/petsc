# $Id: makefile,v 1.25 1997/08/22 15:09:57 bsmith Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = general.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = general.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/impls/seq/general/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

