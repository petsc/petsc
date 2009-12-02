
#requirespackage    'PETSC_HAVE_SPOOLES'
#requiresprecision  double

ALL: lib

CFLAGS   = ${SPOOLES_INCLUDE}
FFLAGS   =
SOURCEC	 = mpiaijspooles.c mpispooles.c
SOURCEF	 =
SOURCEH	 =
OBJSC	 = mpiaijspooles.o mpispooles.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/spooles/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
