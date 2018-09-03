#requirespackage  'PETSC_HAVE_ADIOS'

ALL: lib

CFLAGS    = ${ADIOS_INCLUDE}
FFLAGS    =
SOURCEC   = adios.c
SOURCEF   =
SOURCEH   =
MANSEC    = Sys
SUBMANSEC = Viewer
LIBBASE   = libpetscsys
DIRS      =
LOCDIR    = src/sys/classes/viewer/impls/adios/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
