ALL: lib 
IPETSCDIR = ../../../..

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude  -I../../ -D__DIR__='"ksp/impls/lsqr/"'
SOURCEC  = lsqr.c
SOURCEF  =
WSOURCE = 
OBJSC    = lsqr.o
OBJSF    =
WOBJS    = 
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
