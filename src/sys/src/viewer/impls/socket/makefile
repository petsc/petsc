# $Id: makefile,v 1.47 2001/01/15 21:43:07 bsmith Exp balay $ 

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


