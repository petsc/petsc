PETSC_DIR = ../../../..
ALL: lib 

CFLAGS   = $(PETSC_INCLUDE) $(CONF)  \
           -I../../ -D__DIR__='"ksp/impls/bcgs/"'
SOURCEC  = bcgs.c
SOURCEF  =
OBJSC    = bcgs.o
OBJSF    =
LIBBASE  = libpetscksp

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
