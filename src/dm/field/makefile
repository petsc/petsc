-include ../petscdir.mk
ALL: lib

SOURCEH = ../../../include/petsc/private/dmfieldimpl.h \
          ../../../include/petscdmfield.h
DIRS    = interface impls tutorials
LOCDIR  = src/dm/field/
MANSEC  = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
