IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = 
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = cr bcgs cg cgs gmres cheby rich lsqr preonly tcqmr tfqmr qcg

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
