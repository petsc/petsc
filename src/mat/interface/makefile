IPETSCDIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I..   $(CONF) -D__DIR__='"mat/interface/"'
SOURCEC  = matrix.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = matrix.o 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
