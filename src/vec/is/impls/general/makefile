ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I../../.. -I$(ITOOLSDIR) \
           -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = general.c 
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = general.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)

include $(ITOOLSDIR)/bmake/$(PARCH).site
include $(ITOOLSDIR)/bmake/$(PARCH).$(BOPT)
include $(ITOOLSDIR)/bmake/$(PARCH)
