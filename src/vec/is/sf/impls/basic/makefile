-include ../../../../../../petscdir.mk

SOURCEC       = sfbasic.c sfpack.c sfmpi.c
LIBBASE       = libpetscvec
DIRS          = allgatherv allgather gatherv gather alltoall neighbor cuda hip kokkos nvshmem
MANSEC        = Vec
SUBMANSEC     = PetscSF

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

