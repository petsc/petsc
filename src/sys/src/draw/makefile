#PETSC_DIR = ../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC  =
SOURCEF  =
SOURCEH  = drawimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsys
DIRS     = interface impls examples utils
LOCDIR   = src/draw/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
