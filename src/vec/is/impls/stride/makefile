PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' -I../.. 
SOURCEC  = stride.c
SOURCEF  =
SOURCEH  = 
OBJSC    = stride.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/impls/seq/stride/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

