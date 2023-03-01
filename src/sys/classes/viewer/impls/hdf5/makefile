-include ../../../../../../petscdir.mk
#requirespackage  'PETSC_HAVE_HDF5'

CFLAGS    = ${HDF5_INCLUDE}
SOURCEC   = hdf5v.c
SOURCEF   =
SOURCEH   = ../../../../../../include/petscviewerhdf5.h
MANSEC    = Sys
SUBMANSEC = Viewer
LIBBASE   = libpetscsys
DIRS      =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
