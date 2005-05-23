#requirespackage   'PETSC_HAVE_MATLAB'
#requiresscalar    real
#requiresprecision double

ALL: lib

CFLAGS   = ${MATLAB_INCLUDE}
FFLAGS   =
SOURCEC  = gcreatev.c
SOURCEF  =
SOURCEH  = 
OBJSC    = gcreatev.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/utils/matlab/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

