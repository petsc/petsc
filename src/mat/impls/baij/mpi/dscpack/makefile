
#requirespackage   'PETSC_HAVE_DSCPACK'
#requiresprecision double
#requiresscalar    real

ALL: lib

CFLAGS   = ${DSCPACK_INCLUDE} -DF_NEEDS_UNDSC -DDBL_R_NUM
FFLAGS   =
SOURCEC	 = dscpack.c
SOURCEF	 =
SOURCEH	 = 
OBJSC	 = dscpack.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/baij/mpi/dscpack/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
