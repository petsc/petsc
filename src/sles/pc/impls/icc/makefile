IPETSCDIR = ../../../..

ALL: lib

# This matrix dependency should be fixed!
CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I../.. -I$(IPETSCDIR)/pinclude \
           $(BS_INCLUDE) -I../../../../src/mat \
           $(CONF) -D__DIR__='"pc/impls/icc/"'
SOURCEC  = icc.c iccbs.c
SOURCEF  =
SOURCEH  = icc.h
OBJSC    = icc.o iccbs.o
OBJSF    =
LIBBASE  = libpetscsles
DIRS     = 
MANSEC   = 4

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
