IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCE = 
SOURCEH  = 
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = mpi

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
