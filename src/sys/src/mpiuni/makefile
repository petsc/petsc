PETSC_DIR = ../..

ALL: lib

CFLAGS	 = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC	 = mpi.c
SOURCEF	 = 
SOURCEH	 = 
OBJSC	 = mpi.o
OBJSF	 =
LIBBASE	 = libmpiuni
LOCDIR   = src/mpiuni/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
