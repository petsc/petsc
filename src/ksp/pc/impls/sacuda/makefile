#requirespackage 'PETSC_HAVE_CUDA'
#requirespackage 'PETSC_HAVE_CUSP_DEV'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU  = sacuda.cu
SOURCEF  =
SOURCEH  = 
OBJSC    = sacuda.o
OBJSF    =
LIBBASE  = libpetscksp
DIRS     = 
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/sacuda/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
