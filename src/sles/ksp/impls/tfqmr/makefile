ALL: lib 
PETSC_DIR = ../../../..

CFLAGS   =  $(CONF) $(PETSC_INCLUDE) \
           -I../../ -D__DIR__='"ksp/impls/tfqmr/"'
SOURCEC  = tfqmr.c
SOURCEF  =
OBJSC    = tfqmr.o
OBJSF    =
LIBBASE  = libpetscksp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
