ALL: lib 
PETSC_DIR = ../../../..

CFLAGS   =  $(CONF) $(PETSC_INCLUDE) \
           -I../.. -D__DIR__='"ksp/impls/preonly/"'
SOURCEC  = preonly.c
SOURCEF  =
OBJSC    = preonly.o
OBJSF    =
LIBBASE  = libpetscksp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
