PETSC_DIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I..  -I../.. $(PCONF)\
           $(CONF) -D__DIR__='"pc/impls/mg/"'
SOURCEC  = mg.c fmg.c smg.c mgfunc.c 
SOURCEF  =
SOURCEH  = mgimpl.h
OBJSC    = mg.o fmg.o smg.o mgfunc.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4
LOCDIR   = src/pc/impls/mg/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
