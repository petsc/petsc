#PETSC_DIR = ../../..

ALL: lib

CFLAGS	 = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC	 = iscomp.c
SOURCEF	 =
SOURCEH	 =
OBJSC	 = iscomp.o
OBJSF	 =
LIBBASE	 = libpetscvec
DIRS	 = 
MANSEC	 = 6
LOCDIR	 = src/is/utils/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
