IPETSCDIR = ../..

ALL: lib

CFLAGS   =  -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = snesimpl.h
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsnes
DIRS     = interface impls  examples

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

