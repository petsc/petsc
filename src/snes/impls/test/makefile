#PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = snestest.c
SOURCEF  =
SOURCEH  = 
OBJSC    = snestest.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/test/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

