PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'  -I../..
SOURCEC  = eisen.c
SOURCEF  =
SOURCEH  = 
OBJSC    = eisen.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/eisen/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
