IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../.. -I$(IPETSCDIR)/pinclude\
           $(CONF) -I../../../mat/impls/aij/seq \
           -I../../../mat/impls/row/seq \
           -I../../../mat/impls/bdiag/seq -I../../../mat\
           -D__DIR__='"pc/impls/bjacobi/"'
SOURCEC  = bjacobi.c bjmpiaij.c bjmpirow.c bjmpibd.c
SOURCEF  =
SOURCEH  = bjacobi.h
OBJSC    = bjacobi.o bjmpiaij.o bjmpirow.o bjmpibd.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
