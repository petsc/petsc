
-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_HIP'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEHIP= densehip.hip.cpp
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
MANSEC   = Mat
LOCDIR   = src/mat/impls/dense/seq/hip/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
