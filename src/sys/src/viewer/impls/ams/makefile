# $Id: makefile,v 1.6 1999/02/02 21:21:44 balay Exp bsmith $ 

#requirespackage  '-DPETSC_HAVE_AMS'

ALL: lib 

CFLAGS   = ${AMS_INCLUDE} -DAMS_DIR='"${AMS_DIR}"'
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




