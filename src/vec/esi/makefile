
#requireslanguage  CPP
#requiresscalar    real 
#requiresprecision double
#requirespackage   'PETSC_HAVE_CXX_NAMESPACE'

ALL: lib

CFLAGS   = ${TRILINOS_INCLUDE}
FFLAGS   =
SOURCEC  = eindexspace.c evector.c ecomp.c
OBJSC    = eindexspace.o evector.o ecomp.o
SOURCEH  = ../../../include/esi/petsc/indexspace.h ../../../include/esi/petsc/vector.h
LIBBASE  = libpetscvec
MANSEC   = 
LOCDIR   = src/vec/esi/


include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


