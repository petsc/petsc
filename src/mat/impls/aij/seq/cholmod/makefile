-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_SUITESPARSE'
#requiresprecision double

ALL: lib

CFLAGS   = ${SUITESPARSE_INCLUDE}
FFLAGS   =
SOURCEC  = aijcholmod.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/cholmod/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
