ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../impls -I.. -I$(ITOOLSDIR)/pinclude\
           $(COPT) $(CONF) -D__DIR__='"vec/utils/"'
SOURCEC  = vinv.c gcreate.c vscat.c vpscat.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = vinv.o gcreate.o vscat.o vpscat.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 1

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

runexamples:
