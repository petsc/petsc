IPETSCDIR = ../../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"mat/impls/aij/seq/"'
SOURCEC  = aij.c aijfact.c ij.c
SOURCEF  =
SOURCEH  = aij.h
OBJSC    = aij.o aijfact.o ij.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = order
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
