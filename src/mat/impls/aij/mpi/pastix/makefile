-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_PASTIX'

ALL: lib

CFLAGS   = ${PASTIX_INCLUDE}
FFLAGS   =
SOURCEC	 = pastix.c
SOURCEF	 =
SOURCEH	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/pastix/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
