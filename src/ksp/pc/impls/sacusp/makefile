#requiresdefine 'PETSC_HAVE_CUSP_SMOOTHED_AGGREGATION'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU  = sacusp.cu sacusppoly.cu
SOURCEF  =
SOURCEH  = 
LIBBASE  = libpetscksp
DIRS     = 
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/sacusp/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
