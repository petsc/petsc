#requirespackage 'PETSC_HAVE_KOKKOS_KERNELS'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU =
SOURCEKOKKOS = mpikok.kokkos.cxx
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/mpi/kokkos/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
