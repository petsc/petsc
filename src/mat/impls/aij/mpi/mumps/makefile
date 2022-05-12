-include ../../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_MUMPS'

CFLAGS   = ${MUMPS_INCLUDE}
SOURCEC	 = mumps.c
SOURCEF	 =
SOURCEH	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/mumps/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
