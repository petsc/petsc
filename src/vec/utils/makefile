IPETSCDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../impls -I.. -I$(IPETSCDIR)/pinclude\
           -I$(IPETSCDIR)/src/draw $(COPT) $(CONF) -D__DIR__='"vec/utils/"'
SOURCEC  = vinv.c gcreate.c vscat.c vpscat.c cmesh.c
SOURCEF  =
WSOURCEC = vinvf.c gcreatef.c vscatf.c  cmeshf.c
SOURCEH  = 
OBJSC    = vinv.o gcreate.o vscat.o vpscat.o cmesh.o
WOBJS    = vinvf.o gcreatef.o vscatf.o cmeshf.o
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 1

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

