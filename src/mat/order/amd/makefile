
#requirespackage   'PETSC_HAVE_SUITESPARSE'

ALL: lib

CFLAGS   = ${SUITESPARSE_INCLUDE}
FFLAGS   =
SOURCEC  = amd.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/order/amd/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
