#requirespackage   'PETSC_HAVE_STRUMPACK'

ALL: lib

CFLAGS   = ${STRUMPACK_INCLUDE}
FFLAGS   =
SOURCEC	 = strumpack_mpi.c
SOURCEF	 =
SOURCEH	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/strumpack_mpi/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
