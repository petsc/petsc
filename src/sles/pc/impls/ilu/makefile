IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I../.. -I$(IPETSCDIR)/pinclude\
           $(CONF) -D__DIR__='"pc/impls/ilu/"'
SOURCEC  = ilu.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ilu.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
