PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../..  \
           $(CONF) -D__DIR__='"mat/impls/bdiag/seq/"'
SOURCEC  = bdiag.c bdfact.c
SOURCEF  =
SOURCEH  = bdiag.h
OBJSC    = bdiag.o bdfact.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
