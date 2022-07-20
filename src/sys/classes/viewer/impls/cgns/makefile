##
#requirespackage  'PETSC_HAVE_CGNS'

ALL: lib

CFLAGS    = ${CGNS_INCLUDE}
FFLAGS    =
SOURCEC   = cgnsv.c
SOURCEF   =
SOURCEH   =
MANSEC    = Sys
SUBMANSEC = Viewer
LIBBASE   = libpetscsys
DIRS      =
LOCDIR    = src/sys/classes/viewer/impls/cgns/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
