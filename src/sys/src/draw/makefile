IPETSCDIR = ../..

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCE = 
SOURCEH  = drawimpl.h
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscdraw
DIRS     = src impls examples utils

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
