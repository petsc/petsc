ALL: lib 
IPETSCDIR = ../../../..

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/tfqmr/"'
SOURCEC  = tfqmr.c
SOURCEF  =
WSOURCE = 
OBJSC    = tfqmr.o
OBJSF    =
WOBJS    = 
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
