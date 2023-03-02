-include ../../../../../petscdir.mk
#requirespackage  'PETSC_HAVE_ML'

CFLAGS    = ${ML_INCLUDE}
SOURCEC   = ml.c
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
