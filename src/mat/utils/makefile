IPETSCDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I$(IPETSCDIR)/pinclude\
           -I../impls/aij/seq -I../impls/aij/mpi \
           -I../impls/row/seq -I../impls/row/mpi \
           -I../impls/bdiag/seq -I../impls/bdiag/mpi \
           -I../impls/rowbs/mpi \
           $(CONF) -D__DIR__='"mat/utils/"'
SOURCEC  = gcreate.c convert.c stash.c axpy.c
SOURCEF  =
WSOURCEC = 
SOURCEH  =
OBJSC    = gcreate.o convert.o stash.o axpy.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
DIRS     = complex
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
