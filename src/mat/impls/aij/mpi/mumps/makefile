
#requirespackage   'PETSC_HAVE_MUMPS'
#requiresprecision double

ALL: lib

CFLAGS   = ${MUMPS_INCLUDE}
FFLAGS   =
SOURCEC	 = mumps.c
SOURCEF	 =
SOURCEH	 =
OBJSC	 = mumps.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/mumps/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
