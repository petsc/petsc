PETSC_DIR = ../../..

ALL: lib

CFLAGS   = -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF) $(PCONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = 
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = cr bcgs cg cgs gmres cheby rich lsqr preonly tcqmr tfqmr qcg

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
