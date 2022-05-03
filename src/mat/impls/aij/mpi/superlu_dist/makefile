-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_SUPERLU_DIST'

CFLAGS   = ${SUPERLU_DIST_INCLUDE}
SOURCEC	 = superlu_dist.c
SOURCEF	 =
SOURCEH	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/superlu_dist/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
