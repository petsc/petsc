PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. -I../seq $(PCONF)\
           $(CONF) -D__DIR__='"mat/impls/dense/mpi/"'
SOURCEC  = mpidense.c mmdense.c
SOURCEF  =
SOURCEH  = mpidense.h
OBJSC    = mpidense.o mmdense.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/dense/mpi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
