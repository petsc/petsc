IPETSCDIR = ..

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = 
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = inline is vec ksp sys pc mat sles  draw grid stencil \
           viewer snes da

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

