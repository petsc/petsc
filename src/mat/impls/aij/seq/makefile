ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. \
           -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = ij.c aij.c aijfact.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = ij.o aij.o aijfact.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
DIRS     = order
MANSEC   = 2

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
