# $Id: makefile,v 1.11 2001/01/15 21:43:14 bsmith Exp balay $ 

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



