IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../..  \
          -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"snes/impls/ls/"'
SOURCEC  = ls.c
SOURCEF  =
SOURCEH  = ls.h
OBJSC    = ls.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

