PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' -I../..
SOURCEC  = bjacobi.c
SOURCEF  =
SOURCEH  = bjacobi.h
OBJSC    = bjacobi.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4
LOCDIR   = src/pc/impls/bjacobi/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
