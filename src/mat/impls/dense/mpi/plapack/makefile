
#requirespackage   'PETSC_HAVE_PLAPACK'

ALL: lib

CFLAGS   = ${PLAPACK_INCLUDE}
FFLAGS   =
SOURCEC	 = plapack.c
SOURCEF	 =
SOURCEH	 = 
OBJSC	 = plapack.o 
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/dense/mpi/plapack/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
