ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
           -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = sles.c 
SOURCEF  = 
WSOURCEC = slesf.c
SOURCEH  = 
OBJSC    = sles.o 
WOBJS    = slesf.o
OBJSF    =
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
MANSEC   = 3

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
