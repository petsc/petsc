#PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = asm.c
SOURCEF  =
SOURCEH  = 
OBJSC    = asm.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/asm/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
