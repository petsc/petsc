IPETSCDIR = ../../../..
ALL: lib 

CFLAGS   = $(OPT) $(CONF) $(PETSC_INCLUDE) \
           -I../../ -D__DIR__='"ksp/impls/cgs/"'
SOURCEC  = cgs.c
SOURCEF  =
OBJSC    = cgs.o
OBJSF    = 
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
