#requirespackage 'PETSC_HAVE_TXPETSCGPU'
ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCECU = mpiaijcusparse.cu
SOURCEF	 =
SOURCEH	 = mpicusparsematimpl.h
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/mpicusparse

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
