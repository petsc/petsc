PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' \
	   $(SUPERLU_INCLUDE)
SOURCEC  = baij.c baij2.c baijfact.c dgefa.c dgedi.c dgefa3.c
SOURCEF  =
SOURCEH  = baij.h
OBJSC    = baij.o baij2.o baijfact.o dgefa.o dgedi.o dgefa3.o 
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/baij/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
