ITOOLSDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I../../.. \
           -I$(ITOOLSDIR)/pinclude $(CONF) $(COPT)
SOURCEC  = pvec.c pbvec.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = pvec.o pbvec.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)

include $(ITOOLSDIR)/bmake/$(PARCH)
