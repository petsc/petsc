#requirespackage  'PETSC_HAVE_ADIOS2'

ALL: lib

CFLAGS    = ${ADIOS2_INCLUDE}
FFLAGS    =
SOURCEC   = adios2.c
SOURCEF   =
SOURCEH   =
MANSEC    = Sys
SUBMANSEC = Viewer
LIBBASE   = libpetscsys
DIRS      =
LOCDIR    = src/sys/classes/viewer/impls/adios2/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
