ALL: lib 

PETSC_DIR = ../../../..

CFLAGS   =  $(CONF) $(PETSC_INCLUDE) $(PCONF)\
           -I../.. -D__DIR__='"ksp/impls/cr/"'
FFLAGS   =
SOURCEC  = cr.c 
SOURCEF  = 
OBJSC    = cr.o 
OBJSF    =
LIBBASE  = libpetscksp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
