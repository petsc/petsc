ALL: lib 
#PETSC_DIR = ../../../..

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = tfqmr.c
SOURCEF  =
OBJSC    = tfqmr.o
OBJSF    =
LIBBASE  = libpetscksp
LOCDIR   = src/ksp/impls/tfqmr/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
