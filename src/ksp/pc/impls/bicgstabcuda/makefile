#requirespackage 'PETSC_HAVE_CUSP_SMOOTHED_AGGREGATION'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU  = bicgstabcuda.cu
SOURCEF  =
SOURCEH  = 
OBJSC    = bicgstabcuda.o
OBJSF    =
LIBBASE  = libpetscksp
DIRS     = 
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/bicgstabcuda/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
