
#requireslanguage  CPP
#requiresscalar    real 
#requiresprecision double
#requirespackage   'PETSC_HAVE_CXX_NAMESPACE'

ALL: lib

CFLAGS   = ${TRILINOS_INCLUDE}
FFLAGS   =
SOURCEC  = ematrix.c
OBJSC    = ematrix.o
SOURCEH  = ../../../include/esi/petsc/matrix.h
LIBBASE  = libpetscmat
MANSEC   = 
LOCDIR   = src/mat/esi/


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


