IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../.. -I$(IPETSCDIR)/pinclude\
           $(CONF) -D__DIR__='"pc/impls/lu/"'
SOURCEC  = lu.c
SOURCEF  =
SOURCEH  = 
OBJSC    = lu.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
