#PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = bdiag.c bdfact.c
SOURCEF  =
SOURCEH  = bdiag.h
OBJSC    = bdiag.o bdfact.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/bdiag/seq/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
