ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
          -I$(ITOOLSDIR)/pinclude $(CONF) -D__DIR__='"mg/src/"'
SOURCEC  = mg.c fmg.c smg.c mgfunc.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = mg.o fmg.o smg.o mgfunc.o
OBJSF    =
WSOURCEC = 
WOBJS    = 
LIBBASE  = libpetscpc
LINCLUDE = $(SOURCEH)
MANSEC   = 4

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
