PETSC_DIR = ../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I..\
           -I../impls/aij/seq -I../impls/aij/mpi \
           -I../impls/row/seq -I../impls/row/mpi \
           -I../impls/bdiag/seq -I../impls/bdiag/mpi \
           -I../impls/rowbs/mpi \
           $(CONF) -D__DIR__='"mat/utils/"'
SOURCEC  = gcreate.c convert.c stash.c axpy.c mscat.c zerodiag.c matio.c
SOURCEF  =
SOURCEH  =
OBJSC    = gcreate.o convert.o stash.o axpy.o mscat.o zerodiag.o matio.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
