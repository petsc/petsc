IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = 
OBJSC    =
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = seq mpi

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
