PETSC_DIR = ../../../..
ALL: lib 

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' -I../../ 
SOURCEC  = cgs.c
SOURCEF  =
OBJSC    = cgs.o
OBJSF    = 
LIBBASE  = libpetscksp
LOCDIR   = src/ksp/impls/cgs/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
