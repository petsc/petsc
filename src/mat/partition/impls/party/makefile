ALL: lib

#requirespackage 'PETSC_HAVE_PARTY'

CFLAGS   =
FFLAGS   =
CPPFLAGS = ${PARTY_INCLUDE}
SOURCEC  = party.c
SOURCEH  = 
OBJSC    = party.o
LIBBASE  = libpetscmat
LOCDIR   = src/mat/partition/impls/party/
MANSEC   = MatOrderings

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
