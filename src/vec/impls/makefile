PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) \
           -I../..  $(CONF)
SOURCEC  =  
SOURCEF  =
SOURCEH  = dvecimpl.h dvec2.c bvec1.c
OBJSC    =  
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = seq mpi

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

