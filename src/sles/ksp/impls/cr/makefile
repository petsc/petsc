ALL: lib 

IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude -I../.. -D__DIR__='"ksp/impls/cr/"'
FFLAGS   =
SOURCEC  = cr.c 
SOURCEF  = 
WSOURCEC = 
OBJSC    = cr.o 
WOBJS    =  
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
