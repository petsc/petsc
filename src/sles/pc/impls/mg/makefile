ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  -I../..\
          -I$(ITOOLSDIR)/pinclude $(CONF) -D__DIR__='"pc/impls/mg/"'
SOURCEC  = mg.c fmg.c smg.c mgfunc.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = mg.o fmg.o smg.o mgfunc.o
OBJSF    =
WSOURCEC = 
WOBJS    = 
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
MANSEC   = 4

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
