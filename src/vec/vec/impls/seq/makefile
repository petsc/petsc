
ALL: lib

CFLAGS   = ${MATLAB_INCLUDE}
FFLAGS   =
SOURCEC  = bvec2.c bvec1.c dvec2.c vseqcr.c bvec3.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/seq/
DIRS     = ftn-kernels seqviennacl seqviennaclcuda seqcuda kokkos seqhip

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
