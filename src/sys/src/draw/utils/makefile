IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I../impls -I.. -I$(IPETSCDIR)/pinclude\
           $(COPT) $(CONF) -D__DIR__='"draw/utils/"'
SOURCEC  = axis.c lg.c
SOURCEF  =
SOURCEH  = 
OBJSC    = axis.o lg.o
OBJSF    =
LIBBASE  = libpetscdraw
MANSEC   = x

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
