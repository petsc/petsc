ITOOLSDIR = ../../../../

ALL: lib 
CFLAGS    = $(OPT) -I../.. $(INCLUDEDIRS) -I$(ITOOLSDIR)/pinclude \
            $(XINCLUDEDIR) $(CONF)
SOURCEC   = xinit.c  frame.c wmap.c color.c xops.c text.c
SOURCEF   =
SOURCEH   =
OBJSC     = xinit.o frame.o wmap.o color.o xops.o text.o
OBJSF     =
WSOURCEC  =   
WOBJS     = 
MANSEC    = 7
LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscdraw
DIRS      = 

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
