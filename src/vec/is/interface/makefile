ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I$(ITOOLSDIR) \
           -I$(ITOOLSDIR)/pinclude $(CONF) -D__DIR__='"is/src/"'
SOURCEC  = index.c 
SOURCEF  = 
WSOURCEC = indexf.c
SOURCEH  = 
OBJSC    = index.o 
WOBJS    = indexf.o
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 6

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
