#PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = jacobi.c
SOURCEF  =
SOURCEH  = 
OBJSC    = jacobi.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/jacobi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
