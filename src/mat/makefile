PETSC_DIR = ../..

ALL: lib

CFLAGS   =  -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = matimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = interface impls examples utils

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
