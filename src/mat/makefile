ITOOLSDIR = ../

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I$(ITOOLSDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = src impls tests

include $(ITOOLSDIR)/bmake/$(PARCH).site
include $(ITOOLSDIR)/bmake/$(PARCH).$(BOPT)
include $(ITOOLSDIR)/bmake/$(PARCH)
