#PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' \
	   $(SUPERLU_INCLUDE)
SOURCEC  = aij.c aijfact.c ij.c essl.c dxml.c aijnode.c
SOURCEF  =
SOURCEH  = aij.h
OBJSC    = aij.o aijfact.o ij.o essl.o dxml.o aijnode.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/aij/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
