PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(PCONF) $(PETSC_INCLUDE) -I$(PETSC_DIR)/src/vec
SOURCEC  = vgrid.c
SOURCEF  =
SOURCEH  = vgrid.h
OBJSC    = vgrid.o
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = 

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

