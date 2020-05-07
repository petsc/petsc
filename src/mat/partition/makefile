-include ../../../petscdir.mk
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC   = partition.c spartition.c
SOURCEH   =
LIBBASE   = libpetscmat
LOCDIR    = src/mat/partition/
MANSEC    = Mat
SUBMANSEC = MatOrderings
DIRS      = impls

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
