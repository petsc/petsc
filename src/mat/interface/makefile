#PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = matrix.c mhas.c fdmatrix.c
SOURCEF  =
SOURCEH  = 
OBJSC    = matrix.o mhas.o fdmatrix.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
