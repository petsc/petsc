#PETSC_DIR = ../../../../..

ALL: lib

CFLAGS	 = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC	 = mpibaij.c mbaijfact.c mmbaij.c mpibaijpc.c baijpc.c baijov.c
SOURCEF	 =
SOURCEH	 = mpibaij.h
OBJSC	 = mpibaij.o  mbaijfact.o mmbaij.o mpibaijpc.o baijpc.o baijov.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = 2
LOCDIR	 = src/mat/impls/baij/mpi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
