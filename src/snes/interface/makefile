IPETSCDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
          -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"snes/src/"'
SOURCEC  = snesregi.c snes.c snesj.c
SOURCEF  =
SOURCEH  = 
OBJSC    = snesregi.o snes.o snesj.o
OBJSF    =
WSOURCEC = 
WOBJS    = 
LIBBASE  = libpetscsnes
LINCLUDE = $(SOURCEH)
MANSEC   = n

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

runexamples:

