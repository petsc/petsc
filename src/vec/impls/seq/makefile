ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I../../.. -I$(ITOOLSDIR) $(CONF)
SOURCEC  = dvec3.c bvec2.c 
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = dvec3.o bvec2.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)

include $(ITOOLSDIR)/bmake/$(PARCH).site
include $(ITOOLSDIR)/bmake/$(PARCH).$(BOPT)
include $(ITOOLSDIR)/bmake/$(PARCH)

