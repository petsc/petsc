
#requirespackage  'PETSC_HAVE_HDF4' 
#requiresscalar    real

ALL: lib 

CFLAGS   = ${HDF4_INCLUDE}
FFLAGS   =
SOURCEC  = hdf4v.c
SOURCEF  =
SOURCEH  =
OBJSC    = hdf4v.o
OBJSF    =
MANSEC   = Viewer
LIBBASE  = libpetsc
DIRS     = 
LOCDIR   = src/sys/viewer/impls/hdf4/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
