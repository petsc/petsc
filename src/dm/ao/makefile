#PETSC_DIR = ../..

ALL: lib

CFLAGS	 = $(CPPFLAGS) $(CONF) -D__SDIR__='"$(LOCDIR)"'
SOURCEC	 =
SOURCEF	 =
SOURCEH	 = aoimpl.h
OBJSC	 =
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = interface impls  examples 
LOCDIR   = src/is/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

