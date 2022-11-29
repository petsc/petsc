-include ../../../../../../petscdir.mk

SOURCEH       =
SOURCEC       = sfbasic.c sfpack.c sfmpi.c
SOURCECU      =
LIBBASE       = libpetscvec
DIRS          = allgatherv allgather gatherv gather alltoall neighbor cuda hip kokkos nvshmem
LOCDIR        = src/vec/is/sf/impls/basic/
MANSEC        = Vec
SUBMANSEC     = PetscSF

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

