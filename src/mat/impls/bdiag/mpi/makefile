PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. -I../seq \
           $(CONF)
SOURCEC  = mmbdiag.c mpibdiag.c
SOURCEF  =
SOURCEH  = mpibdiag.h
OBJSC    = mmbdiag.o mpibdiag.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
