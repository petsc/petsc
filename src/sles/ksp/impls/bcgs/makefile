IPETSCDIR = ../../../..
ALL: lib 

CFLAGS   = $(OPT) $(PETSC_INCLUDE) $(CONF)  \
           -I../../ -D__DIR__='"ksp/impls/bcgs/"'
SOURCEC  = bcgs.c
SOURCEF  =
OBJSC    = bcgs.o
OBJSF    =
LIBBASE  = libpetscksp

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
