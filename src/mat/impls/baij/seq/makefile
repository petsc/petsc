PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. $(SUPERLU_INCLUDE) \
           $(CONF) -D__DIR__='"mat/impls/baij/seq/"' $(PCONF)
SOURCEC  = baij.c baijfact.c
SOURCEF  =
SOURCEH  = baij.h
OBJSC    = baij.o baijfact.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/baij/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
