-include ../petscdir.mk
ALL: lib

#requirespackage 'PETSC_HAVE_CHACO'

CFLAGS    =
FFLAGS    =
CPPFLAGS  = ${CHACO_INCLUDE}
SOURCEC   = chaco.c
SOURCEH   =
LIBBASE   = libpetscmat
LOCDIR    = src/mat/partition/impls/chaco/
MANSEC    = Mat
SUBMANSEC = MatOrderings

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
