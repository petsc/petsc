#PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' 
SOURCEC  = ls.c
SOURCEF  =
SOURCEH  = ls.h
OBJSC    = ls.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n
LOCDIR   = src/snes/impls/ls/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

