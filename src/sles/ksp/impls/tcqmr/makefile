ALL: lib 
PETSC_DIR = ../../../..

CFLAGS   =  $(CONF) $(PETSC_INCLUDE) \
           -I../../ -D__DIR__='"ksp/impls/tcqmr/"'
SOURCEC  = tcqmr.c
SOURCEF  =
SOURCEH  = tcqmrp.h
OBJSC    = tcqmr.o
OBJSF    = 
LIBBASE  = libpetscksp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
