# $Id: makefile,v 1.2 2000/11/28 17:32:36 bsmith Exp bsmith $ 

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
LOCDIR   = src/contrib/lahaye/ramg

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
