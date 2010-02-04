
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
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/sys/draw/impls/x/
NOADIC    = true

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${PETSC_DIR}/conf/test
