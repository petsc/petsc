#requirespackage 'PETSC_HAVE_KOKKOS_KERNELS'
-include ../../../../../../petscdir.mk
ALL: lib

CFLAGS        =
FFLAGS        =
SOURCECU      =
SOURCEKOKKOS  = aijkok.kokkos.cxx
SOURCEF       =
SOURCEH       = aijkokkosimpl.hpp
LIBBASE       = libpetscmat
DIRS          =
MANSEC        = Mat
LOCDIR        = src/mat/impls/aij/seq/kokkos/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
