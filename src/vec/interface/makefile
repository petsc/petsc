
PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I..  $(CONF) -D__DIR__='"vec/interface/"' \
           $(PCONF)
SOURCEC  = vector.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = vector.o 
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1
LOCDIR   = src/vec/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

