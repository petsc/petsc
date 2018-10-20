#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCECU = mpicuda.cu
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/mpi/mpicuda/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
