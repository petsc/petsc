# $Id: makefile,v 1.5 2001/01/15 21:43:33 bsmith Exp balay $

#requirespackage 'PETSC_HAVE_WIN32'

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
