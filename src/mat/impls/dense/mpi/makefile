#PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = mpidense.c mmdense.c
SOURCEF  =
SOURCEH  = mpidense.h
OBJSC    = mpidense.o mmdense.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = 
MANSEC   = 2
LOCDIR   = src/mat/impls/dense/mpi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
