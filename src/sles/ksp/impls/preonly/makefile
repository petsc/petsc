ALL: lib 

#PETSC_DIR = ../../../..

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = preonly.c
SOURCEF  =
OBJSC    = preonly.o
OBJSF    =
LIBBASE  = libpetscksp
LOCDIR   = src/ksp/impls/preonly/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
