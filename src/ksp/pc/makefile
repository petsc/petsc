-include ../../../petscdir.mk
ALL: lib

SOURCEH   = ../../../include/petsc/private/pcimpl.h ../../../include/petscpc.h \
            ../../../include/petsc/private/pcgamgimpl.h ../../../include/petscpctypes.h
DIRS      = interface impls tests tutorials
LOCDIR    = src/ksp/pc/
MANSEC    = KSP
SUBMANSEC = PC

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
