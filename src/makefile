PETSC_DIR = ..

CFLAGS   = -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF) $(PCONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = 
OBJSC    =
OBJSF    =
LIBBASE  = libpetscvec
DIRS     = inline is vec ksp sys pc mat sles draw \
           viewer snes da mpiuni contrib benchmarks plog

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

