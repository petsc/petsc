#requirespackage 'PETSC_HAVE_KOKKOS_KERNELS'
-include ../../../../../../petscdir.mk
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC   =
SOURCEKOKKOS = bjkokkos.kokkos.cxx
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
DIRS      =
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/bjacobi/bjkokkos

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
