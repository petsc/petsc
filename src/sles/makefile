IPETSCDIR = ../../

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = slesimpl.h
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = src examples

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
