PETSC_DIR = ../..

ALL: lib

CFLAGS   =  -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF) $(PCONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = daimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = src  examples

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

