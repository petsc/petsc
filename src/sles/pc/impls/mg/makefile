IPETSCDIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I..  -I../..\
           $(CONF) -D__DIR__='"pc/impls/mg/"'
SOURCEC  = mg.c fmg.c smg.c mgfunc.c 
SOURCEF  =
SOURCEH  = mgimpl.h
OBJSC    = mg.o fmg.o smg.o mgfunc.o
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
