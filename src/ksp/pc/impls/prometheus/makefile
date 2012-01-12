
#requirespackage  'PETSC_HAVE_PROMETHEUS'
#requiresprecision double
#requiresscalar    real


ALL: lib

CFLAGS   = ${PROMETHEUS_INCLUDE}
FFLAGS   =
SOURCEC  = petscprom.c
SOURCEF  =
SOURCEH  = 
OBJSC    = petscprom.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/prometheus/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test

