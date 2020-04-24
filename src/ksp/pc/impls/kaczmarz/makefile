-include ../petscdir.mk
ALL: lib

CFLAGS    = ${GAMG_INCLUDE}
FFLAGS    =
SOURCEC   = kaczmarz.c
SOURCEF   =
SOURCEH   =
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/kaczmarz/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
