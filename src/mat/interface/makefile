PETSC_DIR = ../../..

ALL: lib

CFLAGS   = $(PETSC_INCLUDE) -I.. $(PCONF) $(CONF) -D__DIR__='"mat/interface/"'
SOURCEC  = matrix.c mhas.c
SOURCEF  =
SOURCEH  = 
OBJSC    = matrix.o mhas.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/interface/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
