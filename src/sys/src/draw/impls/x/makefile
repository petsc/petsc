IPETSCDIR = ../../../../

ALL: lib 
CFLAGS    = $(OPT) -I../.. $(INCLUDEDIRS) -I$(IPETSCDIR)/pinclude \
            $(X11_INCLUDE) $(CONF) -D__DIR__='"draw/impls/x/"'
SOURCEC   = xinit.c  frame.c wmap.c color.c xops.c text.c tone.c
SOURCEF   =
SOURCEH   = ximpl.h
OBJSC     = xinit.o frame.o wmap.o color.o xops.o text.o tone.o
OBJSF     =
WSOURCEC  =   
WOBJS     = 
MANSEC    = x
LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscdraw
DIRS      = 

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
