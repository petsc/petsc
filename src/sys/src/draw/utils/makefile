ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../impls -I.. -I$(ITOOLSDIR)/pinclude\
           $(COPT) $(CONF) -D__DIR__='"draw/utils/"'
SOURCEC  = axis.c lg.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = axis.o lg.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscdraw
LINCLUDE = $(SOURCEH)
MANSEC   = x

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
