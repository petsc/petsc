IPETSCDIR = ../..

ALL: lib

CFLAGS   =  -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = vecimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = utils interface impls  examples

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

