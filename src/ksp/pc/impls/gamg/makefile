
ALL: lib

CFLAGS   = ${GAMG_INCLUDE}
FFLAGS   =
SOURCEC  = gamg.c agg.c geo.c tools.c
SOURCEF  =
SOURCEH  = ../../../../../include/petscpcmg.h gamg.h 
OBJSC    = gamg.o agg.o geo.o tools.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = PC
LOCDIR   = src/ksp/pc/impls/gamg/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
