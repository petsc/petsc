ITOOLSDIR = ../../

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I$(ITOOLSDIR) \
           -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = index.c 
SOURCEF  = 
WSOURCEC = indexf.c
SOURCEH  = 
OBJSC    = index.o 
WOBJS    = indexf.o
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 6

include $(ITOOLSDIR)/bmake/$(PARCH)
