IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCE = 
SOURCEH  = 
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = jacobi none sor lu shell bjacobi mg eisens ilu icc spai

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
