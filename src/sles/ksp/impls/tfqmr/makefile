ALL: lib 
IPETSCDIR = ../../../..

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/tfqmr/"'
SOURCEC  = tfqmr.c
SOURCEF  =
OBJSC    = tfqmr.o
OBJSF    =
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
