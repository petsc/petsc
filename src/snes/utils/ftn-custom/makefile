#requiresdefine   'PETSC_HAVE_FORTRAN'

ALL: lib

CFLAGS   =
FFLAGS   =
SOURCEC  = zdmdasnesf.c zdmlocalsnesf.c zdmsnesf.c
SOURCEF  =
SOURCEH  =
LIBBASE  = libpetscsnes
MANSEC   = SNES
DIRS     =
EDIRS    =
LOCDIR   = src/snes/utils/ftn-custom/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

