IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) $(PETSC_INCLUDE) \
           -I../..   $(CONF) -D__DIR__='"mat/impls/shell/"'
SOURCEC  = shell.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = shell.o 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
