IPETSCDIR = ../../../

ALL: lib

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
DIRS     = cr bcgs cg cgs gmres cheby rich lsqr preonly tcqmr tfqmr qcg

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
