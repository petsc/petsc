PETSC_DIR = ../..

ALL: lib

CFLAGS   =  -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF) $(PCONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = pcimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = utils interface impls  examples

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
