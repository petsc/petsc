PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = tr.c
SOURCEF  =
SOURCEH  = tr.h
OBJSC    = tr.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/tr/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

