#requirespackage 'PETSC_HAVE_KOKKOS_KERNELS'
-include ../../../../../../petscdir.mk

SOURCECU      =
SOURCEKOKKOS  = aijkok.kokkos.cxx
SOURCEF       =
SOURCEH       = aijkok.hpp
LIBBASE       = libpetscmat
DIRS          =
MANSEC        = Mat
LOCDIR        = src/mat/impls/aij/seq/kokkos/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
