PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. -I../seq \
           $(BS_INCLUDE) $(CONF)
SOURCEC  = mpirowbs.c cholbs.c
SOURCEF  =
SOURCEH  = mpirowbs.h
OBJSC    = mpirowbs.o cholbs.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     =
MANSEC   = 2

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
