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
LOCDIR   = src/mat/impls/dense/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
