IPETSCDIR = ../../

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = drawimpl.h
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscdraw
LINCLUDE = $(SOURCEH)
DIRS     = src impls examples utils

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
