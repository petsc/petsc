-include ../petscdir.mk
#requirespackage 'PETSC_HAVE_OPENCL'

ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC   = feopencl.c
SOURCEF   =
LIBBASE   = libpetscdm
DIRS      =
LOCDIR    = src/dm/dt/fe/impls/opencl/
MANSEC    = DM
SUBMANSEC = FE

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
