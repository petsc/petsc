#requirespackage 'PETSC_HAVE_CUSP'
ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCECU = mpicusp.cu
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/mpi/mpicusp/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
