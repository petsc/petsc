#PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = dense.c 
SOURCEF  =
SOURCEH  = dense.h
OBJSC    = dense.o 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/dense/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
