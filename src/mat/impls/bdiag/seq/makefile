IPETSCDIR = ../../../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I../../..  \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"mat/impls/bdiag/seq/"'
SOURCEC  = bdiag.c bdfact.c
SOURCEF  =
SOURCEH  = bdiag.h
OBJSC    = bdiag.o bdfact.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
