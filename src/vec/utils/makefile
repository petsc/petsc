ITOOLSDIR = ../../

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I$(ITOOLSDIR) $(COPT) $(CONF)
SOURCEC  = gcreate.c 
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = gcreate.o 
WOBJS    = 
OBJSF    =
LIBBASE  = $(COMPLEX)/libpetscvec
LINCLUDE = $(SOURCEH)

include $(ITOOLSDIR)/bmake/$(PARCH).site
include $(ITOOLSDIR)/bmake/$(PARCH).$(BOPT)
include $(ITOOLSDIR)/bmake/$(PARCH)
