IPETSCDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I$(IPETSCDIR)/pinclude\
           -I../impls/aij/seq -I../impls/aij/mpi \
           -I../impls/row/seq -I../impls/row/mpi \
           $(CONF) -D__DIR__='"mat/utils"'
SOURCEC  = gcreate.c convert.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = gcreate.o convert.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
DIRS     = complex
MANSEC   = 2

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
