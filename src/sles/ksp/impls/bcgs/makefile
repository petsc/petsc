PETSC_DIR = ../../../..
ALL: lib 

CFLAGS   = $(PETSC_INCLUDE) $(CONF) $(PCONF) \
           -I../../ -D__DIR__='"ksp/impls/bcgs/"'
SOURCEC  = bcgs.c
SOURCEF  =
OBJSC    = bcgs.o
OBJSF    =
LIBBASE  = libpetscksp
LOCDIR   = src/ksp/impls/bcgs/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
