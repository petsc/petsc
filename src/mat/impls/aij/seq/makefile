ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)  -I$(ITOOLSDIR)/include -I../../.. \
           -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = aij.c 
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = aij.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)

include $(ITOOLSDIR)/bmake/$(ARCH).site
include $(ITOOLSDIR)/bmake/$(ARCH).$(BOPT)
include $(ITOOLSDIR)/bmake/$(ARCH)
