#PETSC_DIR = ../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  =
SOURCEF  =
SOURCEH  = kspimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = interface impls
LOCDIR   = src/ksp/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
