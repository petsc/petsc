ALL: lib 
IPETSCDIR = ../../../..

CFLAGS   =  $(CONF) $(PETSC_INCLUDE) \
            -I../../ -D__DIR__='"ksp/impls/lsqr/"'
SOURCEC  = lsqr.c
SOURCEF  =
OBJSC    = lsqr.o
OBJSF    =
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
