IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"sles/src/"'
SOURCEC  = sles.c 
SOURCEF  = 
WSOURCE = slesf.c
SOURCEH  = 
OBJSC    = sles.o 
WOBJS    = slesf.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 3

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
