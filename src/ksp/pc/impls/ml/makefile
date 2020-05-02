-include ../../../../../petscdir.mk
#requirespackage  'PETSC_HAVE_ML'

ALL: lib

CFLAGS    = ${ML_INCLUDE}
FFLAGS    =
SOURCEC   = ml.c
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/ml/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
