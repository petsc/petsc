#PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"'
SOURCEC  = posindep.c
SOURCEF  =
SOURCEH  = 
OBJSC    = posindep.o
OBJSF    =
LIBBASE  = libpetscts
MANSEC   = t
LOCDIR   = src/ts/impls/pseudo/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base

