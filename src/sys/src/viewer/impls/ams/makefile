# $Id: makefile,v 1.9 2000/08/17 04:50:22 bsmith Exp bsmith $ 

#requirespackage  'PETSC_HAVE_AMS'

ALL: lib 

CFLAGS   = ${AMS_INCLUDE} -DAMS_HOME='"${AMS_HOME}"'
FFLAGS   =
SOURCEC  = ams.c amsopen.c
SOURCEF  =
SOURCEH  =
OBJSC    = ams.o amsopen.o
OBJSF    =
MANSEC   = Viewer
LIBBASE  = libpetsc
DIRS     = 
LOCDIR   = src/sys/src/viewer/impls/ams/

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test



