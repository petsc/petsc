# $Id: makefile,v 1.8 2000/04/09 03:08:56 bsmith Exp bsmith $ 

#requirespackage  '-DPETSC_HAVE_AMS'

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



