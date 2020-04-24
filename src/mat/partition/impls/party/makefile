-include ../petscdir.mk
ALL: lib

#requirespackage 'PETSC_HAVE_PARTY'

CFLAGS    =
FFLAGS    =
CPPFLAGS  = ${PARTY_INCLUDE}
SOURCEC   = party.c
SOURCEH   =
LIBBASE   = libpetscmat
LOCDIR    = src/mat/partition/impls/party/
MANSEC    = Mat
SUBMANSEC = MatOrderings

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
