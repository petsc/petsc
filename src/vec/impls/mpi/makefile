#
#   I'd like to add the dependancies pvec.c depends on pdvec.c ...
#  but can't figure out how.
#
IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I../.. \
           -I$(IPETSCDIR)/pinclude $(CONF) $(COPT) \
           -D__DIR__='"vec/impls/mpi/"'
SOURCEC  = pbvec.c
SOURCEF  =
SOURCEH  = pvecimpl.h pdvec.c pvec2.c
OBJSC    = pbvec.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

