#PETSC_DIR = ../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  =
SOURCEF  =
SOURCEH  = daimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = src  examples utils
LOCDIR   = src/da/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

