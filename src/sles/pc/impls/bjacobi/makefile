PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../.. $(PCONF)\
           $(CONF)  -D__DIR__='"pc/impls/bjacobi/"'
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
