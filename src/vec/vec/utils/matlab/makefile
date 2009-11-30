#requirespackage   'PETSC_HAVE_MATLAB_ENGINE'
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

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

