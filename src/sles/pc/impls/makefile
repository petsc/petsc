PETSC_DIR = ../../..

ALL: lib

CFLAGS   = -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF) $(PCONF)
SOURCEC  =
SOURCEF  =
SOURCEH  = 
OBJSC    =
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = jacobi none sor lu shell bjacobi mg eisens ilu icc asm

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
