-include ../petscdir.mk
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = mpidense.c mmdense.c
SOURCEF  =
SOURCEH  = mpidense.h
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/dense/mpi/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
