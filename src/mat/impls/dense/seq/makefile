IPETSCDIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../..  \
           $(CONF) -D__DIR__='"mat/impls/dense/seq/"'
SOURCEC  = dense.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = dense.o 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
