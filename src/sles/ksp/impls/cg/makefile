ALL: lib 
IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(CONF)  \
           -I$(IPETSCDIR)/pinclude -I../.. -D__DIR__='"ksp/impls/cg"'
FFLAGS   =
SOURCEC  = cg.c cgeig.c
SOURCEF  = 
WSOURCEC = cgeigf.c
OBJSC    = cg.o cgeig.o 
WOBJS    = cgeigf.o 
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
