IPETSCDIR = ../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../impls -I..\
           -I$(IPETSCDIR)/src/draw $(COPT) $(CONF) -D__DIR__='"vec/utils/"'
SOURCEC  = vinv.c gcreatev.c vscat.c vpscat.c cmesh.c vecio.c
SOURCEF  =
SOURCEH  = 
OBJSC    = vinv.o gcreatev.o vscat.o vpscat.o cmesh.o vecio.o
OBJSF    =
LIBBASE  = libpetscvec
MANSEC   = 1

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

