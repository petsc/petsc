IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I..  \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"sles/src/"'
SOURCEC  = sles.c 
SOURCEF  = 
SOURCEH  = 
OBJSC    = sles.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 3

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
