IPETSCDIR = ../../../..
ALL: lib 

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(CONF)  \
           -I$(IPETSCDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/bcgs/"'
SOURCEC  = bcgs.c
SOURCEF  =
OBJSC    = bcgs.o
OBJSF    =
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
