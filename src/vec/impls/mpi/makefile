#
#   I'd like to add the dependancies pvec.c depends on pdvec.c ...
#  but can't figure out how.
#
IPETSCDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I../.. \
           -I$(IPETSCDIR)/pinclude $(CONF) $(COPT) \
           -D__DIR__='"vec/impls/mpi/"'
SOURCEC  = pbvec.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = pbvec.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 1

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

runexamples:
