PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS)  $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = ao.c 
SOURCEF  = 
SOURCEH  = 
OBJSC    = ao.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/ao/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
