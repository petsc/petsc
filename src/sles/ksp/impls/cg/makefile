IPETSCDIR = ../../../..
ALL: lib 

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(CONF)  \
           -I$(IPETSCDIR)/pinclude -I../.. -D__DIR__='"ksp/impls/cg/"'
FFLAGS   =
SOURCEC  = cg.c cgeig.c
SOURCEF  =
SOURCEH  = cgctx.h 
WSOURCE = cgeigf.c
OBJSC    = cg.o cgeig.o 
WOBJS    = cgeigf.o 
OBJSF    =
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
