# $Id: makefile,v 1.2 2000/08/17 04:50:34 bsmith Exp balay $

#requirespackage '-DPETSC_HAVE_WIN32'

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = 
SOURCEC   = win32draw.c 
SOURCEF   = 
SOURCEH   = win32draw.h
OBJSC     = win32draw.o
OBJSF     =
MANSEC    = Draw
LIBBASE   = libpetsc
DIRS      = 
LOCDIR    = src/sys/src/draw/impls/win32/
NOADIC    = true

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
