IPETSCDIR = ../..

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCE = 
SOURCEH  = matimpl.h
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = src impls examples utils

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
