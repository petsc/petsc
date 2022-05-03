-include ../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_PARTY'

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
