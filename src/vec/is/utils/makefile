PETSC_DIR = ../../..

ALL: lib

CFLAGS	 =  $(PETSC_INCLUDE) -I.. $(PCONF) \
	   $(CONF) -D__DIR__='"is/utils/"'
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
