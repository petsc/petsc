IPETSCDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I$(IPETSCDIR) \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"ra/src/"'
SOURCEC  = ra.c 
SOURCEF  = 
WSOURCEC = raf.c
SOURCEH  = 
OBJSC    = ra.o 
WOBJS    = raf.o
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 6

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
