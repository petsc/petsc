ITOOLSDIR = ../../../../

ALL: lib 
CFLAGS    = $(OPT) -I../.. $(INCLUDEDIRS) -I$(ITOOLSDIR)/pinclude \
            $(XINCLUDEDIR) $(CONF) -D__DIR__='"draw/impls/x/"'
SOURCEC   = xinit.c  frame.c wmap.c color.c xops.c text.c
SOURCEF   =
SOURCEH   =
OBJSC     = xinit.o frame.o wmap.o color.o xops.o text.o
OBJSF     =
WSOURCEC  =   
WOBJS     = 
MANSEC    = x
LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscdraw
DIRS      = 

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
