PETSC_DIR = ../..

ALL: lib

CFLAGS	 = $(PETSC_INCLUDE)  $(CONF) -D__DIR__='"mpiuni"' $(PCONF)
SOURCEC	 = mpi.c
SOURCEF	 = 
SOURCEH	 = 
OBJSC	 = mpi.o
OBJSF	 =
LIBBASE	 = libmpiuni

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
