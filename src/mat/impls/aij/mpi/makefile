PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. -I../seq $(CONF) \
            -D__DIR__='"mat/impls/aij/mpi/"' $(PCONF)
SOURCEC  = mpiaij.c mmaij.c maijfact.c mpiaijpc.c aijpc.c
SOURCEF  =
SOURCEH  = 
OBJSC    = mpiaij.o mmaij.o maijfact.o mpiaijpc.o aijpc.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
