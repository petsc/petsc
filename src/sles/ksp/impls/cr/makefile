ALL: lib 

PETSC_DIR = ../../../..

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
FFLAGS   =
SOURCEC  = cr.c 
SOURCEF  = 
OBJSC    = cr.o 
OBJSF    =
LIBBASE  = libpetscksp
LOCDIR   = src/ksp/impls/cr/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
