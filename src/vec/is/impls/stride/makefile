# $Id: makefile,v 1.23 1997/08/22 15:10:00 bsmith Exp balay $

ALL: lib

CFLAGS   =  ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = stride.c
SOURCEF  =
SOURCEH  = 
OBJSC    = stride.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/impls/seq/stride/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

