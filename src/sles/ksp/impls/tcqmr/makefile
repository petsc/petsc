ALL: lib 
PETSC_DIR = ../../../..

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' -I../../
SOURCEC  = tcqmr.c
SOURCEF  =
SOURCEH  = tcqmrp.h
OBJSC    = tcqmr.o
OBJSF    = 
LIBBASE  = libpetscksp
LOCDIR   = src/ksp/impls/tcqmr/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
