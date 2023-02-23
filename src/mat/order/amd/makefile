-include ../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_SUITESPARSE'

CFLAGS    = ${SUITESPARSE_INCLUDE}
SOURCEC   = amd.c
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
SUBMANSEC = MatOrderings

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
