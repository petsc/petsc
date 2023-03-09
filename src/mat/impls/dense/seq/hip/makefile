
-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_HIP'

ALL: lib

SOURCEHIP= densehip.hip.cpp
LIBBASE  = libpetscmat
MANSEC   = Mat

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
