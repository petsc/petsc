ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
           -I$(ITOOLSDIR)/pinclude $(CONF) -D__DIR__='"pc/src/"'
SOURCEC  = precon.c pcset.c pcregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = precon.o pcset.o pcregis.o
OBJSF    =
WSOURCEC = preconf.c
WOBJS    = preconf.o
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
MANSEC   = 4

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
