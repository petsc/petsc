IPETSCDIR = ../../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"mat/impls/aij/seq/"'
SOURCEC  = aij.c aijfact.c ij.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = aij.h
OBJSC    = aij.o aijfact.o ij.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
DIRS     = order
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
