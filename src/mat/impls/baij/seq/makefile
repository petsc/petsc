PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' \
	   -I../../.. $(SUPERLU_INCLUDE)
SOURCEC  = baij.c baijfact.c dgefa.c dgedi.c dgefa3.c
SOURCEF  =
SOURCEH  = baij.h
OBJSC    = baij.o baijfact.o dgefa.o dgedi.o dgefa3.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/baij/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
