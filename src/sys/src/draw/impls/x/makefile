#PETSC_DIR = ../../../..

ALL: lib 
CFLAGS    =  $(CPPFLAGS) $(X11_INCLUDE) $(CONF)  -D__SDIR__='"$(LOCDIR)"'
SOURCEC   = xinit.c  frame.c wmap.c xcolor.c xops.c text.c tone.c
SOURCEF   =
SOURCEH   = ximpl.h
OBJSC     = xinit.o frame.o wmap.o xcolor.o xops.o text.o tone.o
OBJSF     =
MANSEC    = x
LIBBASE   = libpetscsys
DIRS      = 
LOCDIR    = src/draw/impls/x/
NOADIC    = true

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
