IPETSCDIR = ../..

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = drawimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscdraw
DIRS     = interface impls examples utils

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
