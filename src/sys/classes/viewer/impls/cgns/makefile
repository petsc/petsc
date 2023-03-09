##
#requirespackage  'PETSC_HAVE_CGNS'

ALL: lib

CFLAGS    = ${CGNS_INCLUDE}
SOURCEC   = cgnsv.c
MANSEC    = Sys
SUBMANSEC = Viewer
LIBBASE   = libpetscsys

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
