-include ../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_SUITESPARSE'

ALL: lib

CFLAGS    = ${SUITESPARSE_INCLUDE}
FFLAGS    =
SOURCEC   = amd.c
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
SUBMANSEC = MatOrderings
LOCDIR    = src/mat/order/amd/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
