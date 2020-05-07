-include ../../../../../petscdir.mk
#requirespackage  'PETSC_HAVE_PARMS'

ALL: lib

CFLAGS    = ${PARMS_INCLUDE}
FFLAGS    =
SOURCEC   = parms.c
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/parms/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

