PETSC_DIR = ../..

ALL: lib

CFLAGS   =  -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = vecimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = utils interface impls  examples

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

