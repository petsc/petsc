IPETSCDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
          -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"snes/src/"'
SOURCEC  = snes.c snesj.c snesmfj.c snesregi.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = snesregi.o snes.o snesj.o snesmfj.o
OBJSF    =
WSOURCEC = snesf.c snesjf.c snesmfjf.c snesregif.c 
WOBJS    = snesf.o snesjf.o snesmfjf.o snesregif.o 
LIBBASE  = libpetscsnes
LINCLUDE = $(SOURCEH)
MANSEC   = n

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

