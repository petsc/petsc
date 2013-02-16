#requirespackage  'PETSC_HAVE_HDF5'

ALL: lib

CFLAGS   = ${HDF5_INCLUDE}
FFLAGS   =
SOURCEC  = hdf5v.c
SOURCEF  =
SOURCEH  = ../../../../../../include/petscviewerhdf5.h
MANSEC   = Viewer
LIBBASE  = libpetscsys
DIRS     =
LOCDIR   = src/sys/classes/viewer/impls/hdf5/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
