# $Id: makefile,v 1.31 1998/05/29 20:35:27 bsmith Exp bsmith $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"'
SOURCEC  = vector.c map.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vector.o map.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/interface/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

