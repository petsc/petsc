ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
          -I$(ITOOLSDIR)/pinclude $(CONF) -D__DIR__='"snes/src/"'
SOURCEC  = nldregis.c nlregis.c nlcreate.c nlscale.c nlstubs.c
SOURCEF  =
SOURCEH  = 
OBJSC    = nldregis.o nlregis.o nlcreate.o nlscale.o nlstubs.o
OBJSF    =
WSOURCEC = 
WOBJS    = 
LIBBASE  = libpetscsnes
LINCLUDE = $(SOURCEH)
MANSEC   = n

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

runexamples:

