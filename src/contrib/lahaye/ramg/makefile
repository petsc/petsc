# $Id: makefile,v 1.1 2000/11/22 22:25:41 bsmith Exp bsmith $ 

#requirespackage 'PETSC_HAVE_RAMG'
#requireslanguage CONLY
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
LOCDIR   = src/contrib/lahaye

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
