ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I$(ITOOLSDIR) \
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

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
