#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS   = 
FFLAGS   =
SOURCECU = mpiaijcuda.cu
SOURCEF	 =
SOURCEH	 = 
OBJSC	 = mpiaijcuda.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 =
MANSEC	 = Mat
LOCDIR	 = src/mat/impls/aij/mpi/mpicuda

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
