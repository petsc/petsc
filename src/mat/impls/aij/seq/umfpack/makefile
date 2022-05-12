-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_SUITESPARSE'
#requiresprecision double

CFLAGS   = ${SUITESPARSE_INCLUDE}
SOURCEC  = umfpack.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/umfpack/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
