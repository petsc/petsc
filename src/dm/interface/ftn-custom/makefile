-include ../petscdir.mk
#requiresdefine   'PETSC_HAVE_FORTRAN'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = zdmf.c zdmgetf.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscdm
MANSEC   = DM
LOCDIR   = src/dm/interface/ftn-custom/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test


