IPETSCDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I$(IPETSCDIR) \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"da/src/"'
SOURCEC  = da.c da1.c
SOURCEF  = 
WSOURCEC = daf.c da1f.c
SOURCEH  = 
OBJSC    = da.o da1.o
WOBJS    = daf.o da1f.o
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 6

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
