PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = aodebug.c
SOURCEF  =
SOURCEH  = 
OBJSC    = aodebug.o
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = 
LMANSEC  = 6
OCDIR    = src/ao/impls/debug/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

