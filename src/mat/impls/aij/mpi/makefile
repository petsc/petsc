PETSC_DIR = ../../../../..

ALL: lib

CFLAGS	 =  $(PETSC_INCLUDE) -I../../.. -I../seq $(CONF) -I../../../../vec \
	    -D__DIR__='"mat/impls/aij/mpi/"' $(PCONF)
SOURCEC	 = mpiaij.c mmaij.c maijfact.c mpiaijpc.c aijpc.c mpiov.c
SOURCEF	 =
SOURCEH	 = 
OBJSC	 = mpiaij.o mmaij.o maijfact.o mpiaijpc.o aijpc.o mpiov.o
OBJSF	 =
LIBBASE	 = libpetscmat
DIRS	 = 
MANSEC	 = 2
LOCDIR	 = src/mat/impls/aij/mpi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
