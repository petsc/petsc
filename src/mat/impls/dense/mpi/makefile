PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. -I../seq \
           $(CONF) -D__DIR__='"mat/impls/dense/mpi/"'
SOURCEC  = mpidense.c mmdense.c mdensefact.c
SOURCEF  =
SOURCEH  = 
OBJSC    = mpidense.o mmdense.o mdensefact.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
