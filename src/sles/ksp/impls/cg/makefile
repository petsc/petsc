IPETSCDIR = ../../../..
ALL: lib 

CFLAGS   = $(OPT) $(PETSC_INCLUDE) $(CONF)  \
           -I$(IPETSCDIR)/pinclude -I../.. -D__DIR__='"ksp/impls/cg/"'
FFLAGS   =
SOURCEC  = cg.c cgeig.c
SOURCEF  =
SOURCEH  = cgctx.h 
OBJSC    = cg.o cgeig.o 
OBJSF    =
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
