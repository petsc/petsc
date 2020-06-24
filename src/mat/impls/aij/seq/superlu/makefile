-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_SUPERLU'

ALL: lib

CFLAGS   = ${SUPERLU_INCLUDE}
FFLAGS   =
SOURCEC  = superlu.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/superlu/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
