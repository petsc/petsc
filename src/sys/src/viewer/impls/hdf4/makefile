# $Id: makefile,v 1.1 2003/06/09 23:20:49 kai Exp kai $ 

ALL: lib 

#requirespackage  'PETSC_HAVE_HDF4'
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
LOCDIR   = src/sys/src/viewer/impls/hdf4/

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test
