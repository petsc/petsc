-include ../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_CUDA'

SOURCECU  = math2opus.cu math2opusutils.cu
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
LOCDIR    = src/mat/impls/h2opus/cuda/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
