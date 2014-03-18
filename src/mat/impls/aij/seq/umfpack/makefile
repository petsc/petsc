
#requirespackage   'PETSC_HAVE_SUITESPARSE'
#requiresprecision double

ALL: lib

CFLAGS   = ${SUITESPARSE_INCLUDE}
FFLAGS   =
SOURCEC  = umfpack.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/umfpack/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
