-include ../../../../../petscdir.mk
#requiresscalar real

ALL: lib

CPPFLAGS  =
CFLAGS    =
FFLAGS    =
SOURCEC   = bitmask.c comm.c gs.c ivec.c  xxt.c xyt.c tfs.c
SOURCEF   =
SOURCEH   = tfs.h
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC
LOCDIR    = src/ksp/pc/impls/tfs/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
