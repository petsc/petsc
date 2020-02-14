#requirespackage 'PETSC_HAVE_VIENNACL'
#requiresdefine 'PETSC_HAVE_CUDA'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU = vecviennaclcuda.cu
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/seq/seqviennaclcuda/
DIRS     =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

