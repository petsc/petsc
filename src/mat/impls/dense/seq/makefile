PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../..  $(PCONF)\
           $(CONF) -D__DIR__='"mat/impls/dense/seq/"'
SOURCEC  = dense.c 
SOURCEF  =
SOURCEH  = dense.h
OBJSC    = dense.o 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
