IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) \
           -I..   $(CONF) -D__DIR__='"mat/src/"'
SOURCEC  = matrix.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = matrix.o 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
