IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I$(IPETSCDIR) \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"da/src/"'
SOURCEC  = da2.c da1.c
SOURCEF  = 
WSOURCE = da2f.c da1f.c
SOURCEH  = 
OBJSC    = da2.o da1.o
WOBJS    = da2f.o da1f.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
