#requirespackage 'PETSC_HAVE_CUDA'
#requiresprecision single
#requiresscalar    complex
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCECU  = cufft.cu
SOURCEF   = 
SOURCEH   = 
OBJSCU    = cufft.o 
OBJSF     = 
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
LOCDIR    = src/mat/impls/cufft

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
