IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I../.. \
           -I$(IPETSCDIR)/pinclude -I$(IPETSCDIR)/src/is $(CONF) \
           -D__DIR__='"vec/impls/seq/"'
SOURCEC  = bvec2.c 
SOURCEF  =
WSOURCE = bvec2f.c
SOURCEH  = 
OBJSC    = bvec2.o 
WOBJS    = bvec2f.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

