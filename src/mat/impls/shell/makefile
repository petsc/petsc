PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) $(PCONF) \
           -I../..   $(CONF) -D__DIR__='"mat/impls/shell/"'
SOURCEC  = shell.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = shell.o 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
