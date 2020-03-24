-include ../petscdir.mk
#requirespackage   'PETSC_HAVE_CUDA'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU = densecuda.cu
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
MANSEC   = Mat
LOCDIR   = src/mat/impls/dense/seq/cuda

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
