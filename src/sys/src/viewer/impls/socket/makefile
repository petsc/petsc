# $Id: makefile,v 1.45 1999/06/30 22:48:48 bsmith Exp bsmith $ 

#
#
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  = 
SOURCEC   = senddense.c send.c sendsparse.c
SOURCEF   =
SOURCEH   = socket.h 
OBJSC     = senddense.o send.o  sendsparse.o
OBJSF     =
MANSEC    = Viewer
LIBBASE   = libpetsc
DIRS      = matlab
LOCDIR    = src/sys/src/viewer/impls/socket/
NOADIC    = true

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test


