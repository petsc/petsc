-include ../../../../../petscdir.mk

DIRS     = cuda hip
SOURCEC  = dense.c densehdf5.c
SOURCEF  =
SOURCEH  = dense.h
LIBBASE  = libpetscmat
MANSEC   = Mat
LOCDIR   = src/mat/impls/dense/seq/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
