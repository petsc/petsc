ALL: lib

IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(PETSC_INCLUDE) \
           -I$(IPETSCDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/gmres/"'
SOURCEC  = gmres.c borthog.c
SOURCEH  = gmresp.h
SOURCEF  =
OBJSC    = gmres.o borthog.o
OBJSF    = 
LIBBASE  = libpetscksp
MANSEC   = 5

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
