ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../.. -I$(ITOOLSDIR)/pinclude\
           $(CONF) -I../../../mat/impls/aij/seq -I../../../mat\
           -D__DIR__='"pc/impls/bjacobi"'
SOURCEC  = bjacobi.c bjaijmpi.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = bjacobi.o bjaijmpi.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
DIRS     = 
MANSEC   = 4

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
