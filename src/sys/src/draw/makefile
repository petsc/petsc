PETSC_DIR = ../..

ALL: lib

CFLAGS   =  -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = drawimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscdraw
DIRS     = interface impls examples utils

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
