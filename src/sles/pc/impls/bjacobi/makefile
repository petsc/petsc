ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../.. -I$(ITOOLSDIR)/pinclude\
           $(CONF) -I../../../mat/impls/aij/seq -I../../../mat
SOURCEC  = bjacobi.c bjaijmpi.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = bjacobi.o bjaijmpi.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscpc
LINCLUDE = $(SOURCEH)
DIRS     = 
MANSEC   = 4

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
