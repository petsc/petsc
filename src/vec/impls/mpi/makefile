ITOOLSDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I../../.. -I$(ITOOLSDIR)\
           -I$(ITOOLSDIR)/pinclude $(CONF) $(COPT)
SOURCEC  = pvec.c pbvec.c pdvec.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = pvec.o pbvec.o pdvec.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)

include $(ITOOLSDIR)/bmake/$(PARCH)
