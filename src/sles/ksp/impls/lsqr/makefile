ALL: lib 
PETSC_DIR = ../../../..

CFLAGS   =  $(CONF) $(PETSC_INCLUDE) $(PCONF) \
            -I../../ -D__DIR__='"ksp/impls/lsqr/"'
SOURCEC  = lsqr.c
SOURCEF  =
OBJSC    = lsqr.o
OBJSF    =
LIBBASE  = libpetscksp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
