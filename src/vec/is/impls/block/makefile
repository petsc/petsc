PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = block.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = block.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/impls/seq/block/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

