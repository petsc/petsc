IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I.. -I$(IPETSCDIR) \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"da/src/"'
SOURCEC  = da2.c da1.c da3.c
SOURCEF  = 
SOURCEH  = 
OBJSC    = da2.o da1.o da3.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
