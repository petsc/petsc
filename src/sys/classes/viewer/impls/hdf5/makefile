#requirespackage  'PETSC_HAVE_HDF5'

ALL: lib

CFLAGS    = ${HDF5_INCLUDE}
FFLAGS    =
SOURCEC   = hdf5v.c
SOURCEF   =
SOURCEH   = ../../../../../../include/petscviewerhdf5.h
MANSEC    = Sys
SUBMANSEC = Viewer
LIBBASE   = libpetscsys
DIRS      =
LOCDIR    = src/sys/classes/viewer/impls/hdf5/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
