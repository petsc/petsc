ALL: lib 

#PETSC_DIR = ../../../..

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = rich.c richscale.c
SOURCEF  =
SOURCEH  = richctx.h
OBJSC    = rich.o richscale.o
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = 5
LOCDIR   = src/ksp/impls/rich/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
