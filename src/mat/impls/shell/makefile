PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = shell.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = shell.o 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2
LOCDIR   = src/mat/impls/shell/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
