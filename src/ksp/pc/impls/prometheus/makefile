
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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test

