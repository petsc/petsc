ITOOLSDIR = ../../

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I$(ITOOLSDIR) \
          -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = vector.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = vector.o 
OBJSF    =
WSOURCEC = vectorf.c
WOBJS    = vectorf.o
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 2

include $(ITOOLSDIR)/bmake/$(PARCH)
