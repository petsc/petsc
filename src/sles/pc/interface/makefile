IPETSCDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"pc/src/"'
SOURCEC  = precon.c pcset.c pcregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = precon.o pcset.o pcregis.o
OBJSF    =
WSOURCEC = preconf.c pcsetf.c pcregisf.c
WOBJS    = preconf.o pcsetf.o pcregisf.o
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
MANSEC   = 4

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
