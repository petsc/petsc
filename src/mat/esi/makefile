# $Id: makefile,v 1.1 2001/08/30 19:38:52 bsmith Exp bsmith $

#requirespackage  'PETSC_HAVE_ESI'
#requireslanguage  CPP
#requiresscalar    real 
#requiresprecision double

ALL: lib

CFLAGS   = ${ESI_INCLUDE}
FFLAGS   =
SOURCEC  = ematrix.c
OBJSC    = ematrix.o
SOURCEH  = ${PETSC_DIR}/include/esi/petsc/matrix.h
LIBBASE  = libpetscmat
MANSEC   = 
LOCDIR   = src/mat/esi/


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


