PETSC_DIR = ../../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../../.. -I../seq \
           $(CONF)
SOURCEC  = mpiaij.c mmaij.c maijfact.c mpiaijpc.c aijpc.c
SOURCEF  =
SOURCEH  = 
OBJSC    = mpiaij.o mmaij.o maijfact.o mpiaijpc.o aijpc.o
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = order
MANSEC   = 2

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
