IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I$(IPETSCDIR)/pinclude \
           -I../.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =  
SOURCEF  =
WSOURCE = 
SOURCEH  = dvecimpl.h dvec2.c bvec1.c
OBJSC    =  
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = seq mpi

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

