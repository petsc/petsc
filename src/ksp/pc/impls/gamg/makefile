#requirespackage  'PETSC_HAVE_TRIANGLE'
#requiresprecision double
#requiresscalar    real


ALL: lib

CFLAGS   = ${GAMG_INCLUDE}
FFLAGS   =
SOURCEC  = gamg.c gamg.h createProlongation.c
SOURCEF  =
SOURCEH  = ../../../../../include/petscpcmg.h
OBJSC    = gamg.o createProlongation.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/gamg/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
