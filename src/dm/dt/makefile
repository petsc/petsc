-include ../petscdir.mk
ALL: lib

SOURCEH = ../../../include/petscdt.h ../../../include/petscds.h ../../../include/petscdstypes.h
DIRS    = interface tests space dualspace fe fv
LOCDIR  = src/dm/dt/
MANSEC  = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
