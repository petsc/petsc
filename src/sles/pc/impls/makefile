IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = 
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = jacobi none sor lu shell bjacobi mg eisens ilu icc

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
