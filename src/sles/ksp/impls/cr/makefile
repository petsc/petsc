ALL: lib 

IPETSCDIR = ../../../..

CFLAGS   =  $(CONF) $(PETSC_INCLUDE) \
           -I../.. -D__DIR__='"ksp/impls/cr/"'
FFLAGS   =
SOURCEC  = cr.c 
SOURCEF  = 
OBJSC    = cr.o 
OBJSF    =
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
