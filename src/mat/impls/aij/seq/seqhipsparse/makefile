-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_HIP'
ALL: lib

SOURCEHIP = aijhipsparse.hip.cpp aijhipsparseband.hip.cpp
SOURCEH   = hipsparsematimpl.h
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
