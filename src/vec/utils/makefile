ITOOLSDIR = ../../

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I$(ITOOLSDIR)/pinclude\
           $(COPT) $(CONF)
SOURCEC  = vinv.c gcreate.c 
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = vinv.o gcreate.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = complex

include $(ITOOLSDIR)/bmake/$(PARCH)
