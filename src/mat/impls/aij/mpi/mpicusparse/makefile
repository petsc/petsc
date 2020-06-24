-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU = mpiaijcusparse.cu
SOURCEF	 =
SOURCEH	 = mpicusparsematimpl.h
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/mpicusparse/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
