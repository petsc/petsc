ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../impls -I.. -I$(ITOOLSDIR)/pinclude\
           $(COPT) $(CONF)
SOURCEC  = axis.c lg.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = axis.o lg.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = x

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
