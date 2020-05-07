-include ../../../petscdir.mk
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC   = coarsen.c scoarsen.c
SOURCEH   =
LIBBASE   = libpetscmat
LOCDIR    = src/mat/coarsen/
MANSEC    = Mat
SUBMANSEC = MatOrderings
DIRS      = impls

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
