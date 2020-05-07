-include ../../../../../petscdir.mk
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC   = mg.c fmg.c smg.c mgfunc.c
SOURCEF   =
SOURCEH   = ../../../../../include/petsc/private/pcmgimpl.h
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/mg/
DIRS      =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
