#requirespackage 'PETSC_HAVE_KOKKOS_KERNELS'
-include ../../../../../petscdir.mk

SOURCEKOKKOS = landau.kokkos.cxx
SOURCECU =
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetscts
MANSEC   = TS
SUBMANSEC= LANDAU
LOCDIR   = src/ts/utils/dmplexlandau/kokkos

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

