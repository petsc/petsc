# $Id: makefile,v 1.12 2001/01/23 20:52:31 balay Exp bsmith $ 

#requirespackage  'PETSC_HAVE_AMS'

ALL: lib 

CFLAGS   = -DAMS_HOME='"${AMS_HOME}"'
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



