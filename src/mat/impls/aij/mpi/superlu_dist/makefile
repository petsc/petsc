
#requirespackage   'PETSC_HAVE_SUPERLU_DIST'
#requiresprecision double

ALL: lib

CFLAGS   = ${SUPERLU_DIST_INCLUDE}
FFLAGS   =
SOURCEC	 = superlu_dist.c
SOURCEF	 =
SOURCEH	 = 
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/superlu_dist/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
