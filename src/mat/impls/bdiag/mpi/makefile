#PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = mmbdiag.c mpibdiag.c mpibdpc.c
SOURCEF  =
SOURCEH  = mpibdiag.h
OBJSC    = mmbdiag.o mpibdiag.o mpibdpc.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/bdiag/mpi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
