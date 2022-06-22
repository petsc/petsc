-include ../../../../../../../petscdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'
#requirespackage 'PETSC_HAVE_MKL_SPARSE_OPTIMIZE'

SOURCEC  = zmpibaijmklf.c
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetscmat
LOCDIR   = src/mat/impls/baij/mpi/baijmkl/ftn-custom/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
