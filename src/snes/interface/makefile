IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
          -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"snes/src/"'
SOURCEC  = snes.c snesj.c snesmfj.c snesregi.c snesut.c
SOURCEF  =
SOURCEH  = 
OBJSC    = snesregi.o snes.o snesj.o snesmfj.o snesut.o
OBJSF    =
WSOURCE  = snesf.c snesjf.c snesmfjf.c snesregif.c snesutf.c
WOBJS    = snesf.o snesjf.o snesmfjf.o snesregif.o snesutf.o
LIBBASE  = libpetscsnes
MANSEC   = n

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

