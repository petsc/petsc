#requirespackage  'PETSC_HAVE_SOCKET'
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

include ${PETSC_DIR}/bmake/common/base
include ${PETSC_DIR}/bmake/common/test


