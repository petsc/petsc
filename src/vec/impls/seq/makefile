IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I.. -I../.. \
           -I$(IPETSCDIR)/pinclude -I$(IPETSCDIR)/src/is $(CONF) \
           -D__DIR__='"vec/impls/seq/"'
SOURCEC  = bvec2.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = bvec2.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

