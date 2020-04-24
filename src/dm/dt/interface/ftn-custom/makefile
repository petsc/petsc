-include ../petscdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = zdtf.c zdtfef.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscdm
MANSEC   = DM
LOCDIR   = src/dm/dt/interface/ftn-custom/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test


