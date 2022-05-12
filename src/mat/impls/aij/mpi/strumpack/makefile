-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_STRUMPACK'

CFLAGS   = ${STRUMPACK_INCLUDE}
SOURCEC	 = strumpack.c
SOURCEF	 =
SOURCEH	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/strumpack/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
