
PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) -I.. $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = vector.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = vector.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

