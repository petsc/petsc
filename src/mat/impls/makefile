IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = 
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = dense aij shell row rowbs bdiag

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
