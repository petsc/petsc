ALL: lib 
IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude  -I../../ -D__DIR__='"ksp/impls/tcqmr/"'
SOURCEC  = tcqmr.c
SOURCEF  =
WSOURCEC = 
OBJSC    = tcqmr.o
OBJSF    = 
WOBJS    = 
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
