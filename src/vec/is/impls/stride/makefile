ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I$(ITOOLSDIR) \
           -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = stride.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = stride.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 6

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
