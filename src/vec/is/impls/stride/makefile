# $Id: makefile,v 1.24 1998/03/19 02:04:45 balay Exp balay $

ALL: lib

CFLAGS   =  ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = stride.c
SOURCEF  =
SOURCEH  = 
OBJSC    = stride.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/impls/seq/stride/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

