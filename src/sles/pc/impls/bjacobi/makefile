IPETSCDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../.. -I$(IPETSCDIR)/pinclude\
           $(CONF) -I../../../mat/impls/aij/seq -I../../../mat\
           -D__DIR__='"pc/impls/bjacobi"'
SOURCEC  = bjacobi.c bjaijmpi.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = bjacobi.h
OBJSC    = bjacobi.o bjaijmpi.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
DIRS     = 
MANSEC   = 4

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
