IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) \
           -I../..  $(CONF)
SOURCEC  =  
SOURCEF  =
SOURCEH  = dvecimpl.h dvec2.c bvec1.c
OBJSC    =  
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = seq mpi

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

