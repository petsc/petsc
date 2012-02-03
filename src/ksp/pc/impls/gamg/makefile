
ALL: lib

CFLAGS   = ${GAMG_INCLUDE}
FFLAGS   =
SOURCEC  = gamg.c sa.c gmg.c
SOURCEF  =
SOURCEH  = ../../../../../include/petscpcmg.h gamg.h 
OBJSC    = gamg.o sa.o gmg.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/gamg/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
