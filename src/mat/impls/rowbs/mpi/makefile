IPETSCDIR = ../../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I../seq \
           -I$(IPETSCDIR)/pinclude $(BS_INCLUDE) $(CONF)
SOURCEC  = mpirowbs.c cholbs.c
SOURCEF  =
SOURCEH  = mpirowbs.h
OBJSC    = mpirowbs.o cholbs.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
