#PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = shell.c shellcnv.c
SOURCEF  =
SOURCEH  = 
OBJSC    = shell.o shellcnv.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/shell/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
