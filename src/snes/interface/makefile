ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
          -I$(ITOOLSDIR)/pinclude $(CONF) -D__DIR__='"vec/src/"'
SOURCEC  = vector.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = vector.o 
OBJSF    =
WSOURCEC = vectorf.c
WOBJS    = vectorf.o
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 1

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

runexamples:

