# $Id: makefile,v 1.39 2000/08/17 19:56:38 balay Exp bsmith $

#requirespackage 'PETSC_HAVE_X11'

ALL: lib 

CFLAGS    =
FFLAGS    =
CPPFLAGS  = ${X11_INCLUDE} 
SOURCEC   = xinit.c  frame.c wmap.c xcolor.c xops.c text.c tone.c drawopenx.c
SOURCEF   =
SOURCEH   = ximpl.h
OBJSC     = xinit.o frame.o wmap.o xcolor.o xops.o text.o tone.o drawopenx.o
OBJSF     =
MANSEC    = Draw
LIBBASE   = libpetsc
DIRS      = 
LOCDIR    = src/sys/src/draw/impls/x/
NOADIC    = true

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
