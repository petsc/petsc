ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
          -I$(ITOOLSDIR)/pinclude $(CONF) -D__DIR__='"draw/src/"'
SOURCEC  = draw.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = draw.o 
OBJSF    =
WSOURCEC = drawf.c
WOBJS    = drawf.o
LIBBASE  = libpetscdraw
LINCLUDE = $(SOURCEH)
MANSEC   = x

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
