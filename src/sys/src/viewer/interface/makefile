IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
          -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"viewer/src/"'
SOURCEC  = view.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = view.o 
OBJSF    =
LIBBASE  = libpetscsys
MANSEC   = x

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

