#PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = bvec2.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = bvec2.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/impls/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

