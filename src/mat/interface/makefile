ITOOLSDIR = ../../

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I$(ITOOLSDIR) $(CONFXS)
SOURCEC  = matrix.c 
SOURCEF  =
WSOURCEC = matrixf.c
SOURCEH  = 
OBJSC    = matrix.o 
WOBJS    = matrixf.o
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)

include $(ITOOLSDIR)/bmake/$(ARCH).site
include $(ITOOLSDIR)/bmake/$(ARCH).$(BOPT)
include $(ITOOLSDIR)/bmake/$(ARCH)
