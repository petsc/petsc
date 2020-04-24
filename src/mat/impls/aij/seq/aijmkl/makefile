-include ../petscdir.mk
#requirespackage 'PETSC_HAVE_MKL_SPARSE'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = aijmkl.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/aij/seq/aijmkl/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
