#PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = shellpc.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = shellpc.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4
LOCDIR   = src/pc/impls/shell/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
