IPETSCDIR = ../..

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = pcimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = utils src impls  examples

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
