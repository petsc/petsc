IPETSCDIR = ../..

ALL: lib

CFLAGS   =  -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = slesimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = interface examples

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
