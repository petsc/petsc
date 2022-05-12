-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_CUDA'

SOURCECU  = aijcusparse.cu aijcusparseband.cu
SOURCEF   =
SOURCEH   = cusparsematimpl.h
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
LOCDIR    = src/mat/impls/aij/seq/seqcusparse/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
