IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I$(IPETSCDIR) \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"is/src/"'
SOURCEC  = index.c 
SOURCEF  = 
WSOURCE = indexf.c
SOURCEH  = 
OBJSC    = index.o 
WOBJS    = indexf.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 6

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
