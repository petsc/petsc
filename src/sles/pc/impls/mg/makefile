IPETSCDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  -I../..\
          -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"pc/impls/mg/"'
SOURCEC  = mg.c fmg.c smg.c mgfunc.c 
SOURCEF  =
SOURCEH  = mgimpl.h
OBJSC    = mg.o fmg.o smg.o mgfunc.o
OBJSF    =
WSOURCEC = 
WOBJS    = 
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
MANSEC   = 4

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
