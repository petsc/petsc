ALL: lib 
IPETSCDIR = ../../../..

CFLAGS   = $(OPT) $(CONF) $(PETSC_INCLUDE) -I../..  \
           -D__DIR__='"ksp/impls/rich/"'
SOURCEC  = rich.c
SOURCEF  =
SOURCEH  = richctx.h
OBJSC    = rich.o 
OBJSF    =
LIBBASE  = libpetscksp
MANSEC   = 5

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
