IPETSCDIR = ../../../..

ALL: lib 
CFLAGS    = -I../.. $(PETSC_INCLUDE)  \
            $(X11_INCLUDE) $(CONF) -D__DIR__='"draw/impls/x/"'
SOURCEC   = xinit.c  frame.c wmap.c color.c xops.c text.c tone.c
SOURCEF   =
SOURCEH   = ximpl.h
OBJSC     = xinit.o frame.o wmap.o color.o xops.o text.o tone.o
OBJSF     =
MANSEC    = x
LIBBASE   = libpetscdraw
DIRS      = 

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
