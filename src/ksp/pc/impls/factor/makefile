-include ../../../../../petscdir.mk
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEC   = factor.c factimpl.c
SOURCEH   = factor.h
LIBBASE   = libpetscksp
DIRS      = lu ilu icc cholesky
LOCDIR    = src/ksp/pc/impls/factor/
MANSEC    = KSP
SUBMANSEC = PC

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

