PETSC_DIR = ../../../..

ALL: lib

CFLAGS   = $(CPPFLAGS) $(CONF) -D__DIR__='"$(LOCDIR)"' -I.. -I../.. 
SOURCEC  = mg.c fmg.c smg.c mgfunc.c 
SOURCEF  =
SOURCEH  = mgimpl.h
OBJSC    = mg.o fmg.o smg.o mgfunc.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4
LOCDIR   = src/pc/impls/mg/

include $(PETSC_DIR)/bmake/$(PETSC_ARCH)/base
