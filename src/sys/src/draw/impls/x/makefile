# $Id: makefile,v 1.40 2000/10/24 20:24:24 bsmith Exp bsmith $

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
MANSEC    = PetscDraw
LIBBASE   = libpetsc
DIRS      = 
LOCDIR    = src/sys/src/draw/impls/x/
NOADIC    = true

include ${PETSC_DIR}/bmake/${PETSC_ARCH}/base
include ${PETSC_DIR}/bmake/common_test
