IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../.. -I$(IPETSCDIR)/pinclude\
           $(CONF) -D__DIR__='"pc/impls/jacobi/"'
SOURCEC  = jacobi.c
SOURCEF  =
SOURCEH  = 
OBJSC    = jacobi.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
