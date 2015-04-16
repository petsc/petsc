#requirespackage 'PETSC_HAVE_CUSP'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU = mpiaijcusp.cu mpiaijAssemble.cu
SOURCEF	 =
SOURCEH	 = mpicuspmatimpl.h
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/mpicusp/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
