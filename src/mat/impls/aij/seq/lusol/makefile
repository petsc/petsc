-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_LUSOL'
#requiresscalar    real
#requiresprecision double

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = lusol.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/lusol/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
