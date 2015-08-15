#requirespackage  'PETSC_HAVE_ML'
#requiresprecision double
#requiresscalar    real


ALL: lib

CFLAGS   = ${ML_INCLUDE}
FFLAGS   =
SOURCEC  = ml.c
SOURCEF  =
SOURCEH  = 
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/ml/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
