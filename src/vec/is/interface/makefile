#PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = index.c 
SOURCEF  = 
SOURCEH  = 
OBJSC    = index.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6
LOCDIR   = src/is/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
