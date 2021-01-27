#requirespackage 'PETSC_HAVE_CUDA'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = curand.c
SOURCECU = curand2.cu
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscsys
DIRS     =
LOCDIR   = src/sys/classes/random/impls/curand/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
