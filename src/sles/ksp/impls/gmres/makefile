ALL: lib

IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/gmres/"'
SOURCEC  = gmres.c borthog.c
WSOURCEC = gmresf.c  
SOURCEH  = gmresp.h
SOURCEF  =
OBJSC    = gmres.o borthog.o
WOBJS    = gmresf.o 
OBJSF    = 
LIBBASE  = libpetscksp
LINCLUDE = 
MANSEC   = 5

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
