ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
          -I$(ITOOLSDIR)/pinclude $(CONF) -D__DIR__='"snes/src/"'
SOURCEC  = snesregi.c snes.c
SOURCEF  =
SOURCEH  = 
OBJSC    = snesregi.o snes.o
OBJSF    =
WSOURCEC = 
WOBJS    = 
LIBBASE  = libpetscsnes
LINCLUDE = $(SOURCEH)
MANSEC   = n

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

runexamples:

