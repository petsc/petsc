PETSC_DIR = ../../../..
ALL: lib 

CFLAGS   =  $(CONF) $(PETSC_INCLUDE) $(PCONF)\
           -I../../ -D__DIR__='"ksp/impls/cgs/"'
SOURCEC  = cgs.c
SOURCEF  =
OBJSC    = cgs.o
OBJSF    = 
LIBBASE  = libpetscksp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
