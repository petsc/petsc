# $Id: makefile,v 1.26 1998/03/19 02:04:44 balay Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = general.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = general.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/impls/seq/general/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

