PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' -I../..
SOURCEC  = aodebug.c
SOURCEF  =
SOURCEH  = 
OBJSC    = aodebug.o
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = 
LOCDIR   = src/ao/impls/debug/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

