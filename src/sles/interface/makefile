IPETSCDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"sles/src/"'
SOURCEC  = sles.c 
SOURCEF  = 
WSOURCEC = slesf.c
SOURCEH  = 
OBJSC    = sles.o 
WOBJS    = slesf.o
OBJSF    =
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
MANSEC   = 3

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
