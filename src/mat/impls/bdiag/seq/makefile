IPETSCDIR = ../../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../..  \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"mat/impls/bdiag/seq/"'
SOURCEC  = bdiag.c
SOURCEF  =
WSOURCE = 
SOURCEH  = bdiag.h
OBJSC    = bdiag.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
