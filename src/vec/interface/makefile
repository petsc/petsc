# $Id: makefile,v 1.29 1998/03/19 02:04:57 balay Exp balay $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = vector.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = vector.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

