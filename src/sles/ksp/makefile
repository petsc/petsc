IPETSCDIR = ../..

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCE = 
SOURCEH  = kspimpl.h
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = src impls  examples

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
