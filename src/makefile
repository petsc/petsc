IPETSCDIR = ../

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = inline is vec ksp sys pc mat sles  draw grid stencil \
           viewer snes da

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

