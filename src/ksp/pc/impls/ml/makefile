#requirespackage  'PETSC_HAVE_ML'
#requiresprecision double
#requiresscalar    real


ALL: lib

CFLAGS   = ${ML_INCLUDE}
FFLAGS   =
SOURCEC  = ml.c
SOURCEF  =
SOURCEH  = ../../../../../include/petscpcmg.h
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/ml/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
