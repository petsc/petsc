# $Id: makefile,v 1.7 1998/03/19 02:04:46 balay Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = block.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = block.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/impls/seq/block/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

