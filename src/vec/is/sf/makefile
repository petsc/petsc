-include ../../../../petscdir.mk

ALL: lib

SOURCEH	  = ../../../../include/petsc/private/sfimpl.h ../../../../include/petscsf.h ../../../../include/petscsftypes.h
SOURCEC   =
LIBBASE	  = libpetscvec
DIRS	  = interface impls utils tests tutorials
LOCDIR    = src/vec/is/sf/
MANSEC    = Vec
SUBMANSEC = PetscSF

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

