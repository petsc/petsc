#PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = umtr.c
SOURCEF  =
SOURCEH  = umtr.h
OBJSC    = umtr.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/umtr/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

