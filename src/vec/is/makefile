PETSC_DIR = ../..

ALL: lib

CFLAGS	 =  -I$(PETSC_DIR)/include -I.. -I$(PETSC_DIR) $(CONF) $(PCONF)
SOURCEC	 =
SOURCEF	 =
SOURCEH	 = isimpl.h
OBJSC	 =
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = interface impls  examples utils

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

