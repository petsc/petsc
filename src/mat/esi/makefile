# $Id: makefile,v 1.2 2001/09/07 20:10:43 bsmith Exp bsmith $

#requireslanguage  CPP
#requiresscalar    real 
#requiresprecision double
#requirespackage   'PETSC_HAVE_NAMESPACE'

ALL: lib

CFLAGS   = ${CCA_INCLUDE}  ${TRILINOS_INCLUDE}
FFLAGS   =
SOURCEC  = ematrix.c
OBJSC    = ematrix.o
SOURCEH  = ../../../include/esi/petsc/matrix.h
LIBBASE  = libpetscmat
MANSEC   = 
LOCDIR   = src/mat/esi/


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


