# $Id: makefile,v 1.27 1998/04/23 23:32:41 balay Exp bsmith $

ALL: lib

CFLAGS   = ${CPPFLAGS} -D__SDIR__='"${LOCDIR}"' 
SOURCEC  = general.c strgen.c
SOURCEF  =
SOURCEH  = general.h
OBJSC    = general.o strgen.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/impls/seq/general/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base

