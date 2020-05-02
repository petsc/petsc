-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_MKL_SPARSE_OPTIMIZE'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = baijmkl.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = Mat
LOCDIR   = src/mat/impls/baij/seq/baijmkl/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
