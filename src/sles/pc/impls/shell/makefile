IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) -I$(IPETSCDIR)/pinclude \
           -I../..   $(CONF) -D__DIR__='"pc/impls/shell/"'
SOURCEC  = shellpc.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = shellpc.o 
OBJSF    =
LIBBASE  = libpetscsles
MANSEC   = 4

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
