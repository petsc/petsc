IPETSCDIR = ../..

ALL: lib

CFLAGS   =  -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = isimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = interface impls  examples

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

