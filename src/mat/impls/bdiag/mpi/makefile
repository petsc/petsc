PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. -I../seq \
           $(CONF) -D__DIR__='"mat/impls/bdiag/mpi/"'
SOURCEC  = mmbdiag.c mpibdiag.c mpibdpc.c
SOURCEF  =
SOURCEH  = mpibdiag.h
OBJSC    = mmbdiag.o mpibdiag.o mpibdpc.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
