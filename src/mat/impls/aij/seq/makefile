PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. $(SUPERLU_INCLUDE) \
           $(CONF) -D__DIR__='"mat/impls/aij/seq/"' $(PCONF)
SOURCEC  = aij.c aijfact.c ij.c superlu.c essl.c dxml.c aijnode.c
SOURCEF  =
SOURCEH  = aij.h
OBJSC    = aij.o aijfact.o ij.o superlu.o essl.o dxml.o aijnode.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/aij/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
