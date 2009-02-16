#requirespackage  'PETSC_HAVE_HDF5'
#requiresscalar    real

ALL: lib 

CFLAGS   = ${HDF5_INCLUDE}
FFLAGS   =
SOURCEC  = hdf5v.c
SOURCEF  =
SOURCEH  =
OBJSC    = hdf5v.o
OBJSF    =
MANSEC   = Viewer
LIBBASE  = libpetsc
DIRS     = ftn-custom
LOCDIR   = src/sys/viewer/impls/hdf5/

include ${PETSC_DIR}/conf/base
include ${PETSC_DIR}/conf/test
