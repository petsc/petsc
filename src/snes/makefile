PETSC_DIR = ../..

ALL: lib

CFLAGS   =  -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF) $(PCONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = snesimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsnes
DIRS     = interface impls  examples

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

