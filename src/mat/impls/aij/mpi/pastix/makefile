
#requirespackage   'PETSC_HAVE_PASTIX'
#requiresprecision double

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

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
