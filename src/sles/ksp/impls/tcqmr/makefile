ALL: lib 
IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude  -I../../ -D__DIR__='"ksp/impls/tcqmr/"'
SOURCEC  = tcqmr.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = tcqmrp.h
OBJSC    = tcqmr.o
OBJSF    = 
WOBJS    = 
LIBBASE  = libpetscksp
LINCLUDE = 

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
