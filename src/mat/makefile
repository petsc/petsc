IPETSCDIR = ../..

ALL: lib

CFLAGS   =  -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = matimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = interface impls examples utils

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
