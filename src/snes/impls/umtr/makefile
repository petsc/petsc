IPETSCDIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../..  \
          $(CONF) -I$(IPETSCDIR)/src/ksp \
          -I$(IPETSCDIR)/src/ksp/impls/qcg -D__DIR__='"snes/impls/umtr/"'
SOURCEC  = umtr.c
SOURCEF  =
SOURCEH  = umtr.h
OBJSC    = umtr.o
OBJSF    =
LIBBASE  = libpetscsnes
MANSEC   = n

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

