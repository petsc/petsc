# $Id: makefile,v 1.6 1997/08/22 15:10:03 bsmith Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} ${CONF} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = block.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = block.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/impls/seq/block/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

