PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. \
           $(CONF) -D__DIR__='"mat/impls/aij/seq/"'
SOURCEC  = aij.c aijfact.c ij.c
SOURCEF  =
SOURCEH  = aij.h
OBJSC    = aij.o aijfact.o ij.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
