#requirespackage 'PETSC_HAVE_VIENNACL'
#requiresdefine 'PETSC_HAVE_CUDA'

ALL: lib

CFLAGS    =
FFLAGS    =
SOURCECU  = aijviennaclcuda.cu
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
LOCDIR    = src/mat/impls/aij/seq/seqviennaclcuda/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
