# $Id: makefile,v 1.3 2001/01/15 21:49:37 bsmith Exp bsmith $ 

#requirespackage 'PETSC_HAVE_RAMG'
#requiresscalar real 

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = petscfunc.c  ramgpetsc.c
SOURCEF  = 
SOURCEH  = petscfunc.h  ramgfunc.h
OBJSC    = petscfunc.o  ramgpetsc.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = PC
LOCDIR   = src/contrib/lahaye/ramg

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
