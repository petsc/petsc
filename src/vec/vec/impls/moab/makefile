#requirespackage  'PETSC_HAVE_MOAB'
#requiresprecision double
#requiresscalar    real
#requireslanguage  CXXONLY

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = vecmoab.cxx
SOURCEF  =
SOURCEH  = vecmoabimpl.h
OBJSC    = $(SOURCEC:.c=.o)
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/moab/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

