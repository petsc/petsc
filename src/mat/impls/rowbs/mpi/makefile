PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. -I../seq $(PCONF)\
           $(BS_INCLUDE) $(CONF) -D__DIR__='"mat/impls/rowbs/mpi/"'
SOURCEC  = mpirowbs.c cholbs.c ilubs.c iccbs.c
SOURCEF  =
SOURCEH  = mpirowbs.h
OBJSC    = mpirowbs.o cholbs.o ilubs.o iccbs.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = 2
LOCDIR   = src/mat/impls/rowbs/mpi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
