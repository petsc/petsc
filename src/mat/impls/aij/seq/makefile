IPETSCDIR = ../../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"mat/impls/aij/seq/"'
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

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
