PETSC_DIR = ../../..

ALL: lib

CFLAGS   =  -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF) $(PCONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = 
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = dense aij shell rowbs bdiag order baij

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
