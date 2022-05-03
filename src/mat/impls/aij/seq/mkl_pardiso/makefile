-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_MKL_PARDISO'

CFLAGS   = ${MKL_PARDISO_INCLUDE}
SOURCEC	 = mkl_pardiso.c mkl_utils.c
SOURCEF	 =
SOURCEH	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/seq/mkl_pardiso/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
