IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I..  \
          -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"draw/src/"'
SOURCEC  = draw.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = draw.o 
OBJSF    =
LIBBASE  = libpetscdraw
MANSEC   = x

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
