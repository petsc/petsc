#requirespackage   'PETSC_HAVE_MKL_PARDISO'

ALL: lib

CFLAGS   = ${MKL_PARDISO_INCLUDE}
FFLAGS   =
SOURCEC	 = pardiso.c
SOURCEF	 =
SOURCEH	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/seq/pardiso/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
