PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I.. $(PCONF) $(CONF) -D__DIR__='"mat/interface/"'
SOURCEC  = matrix.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = matrix.o 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
