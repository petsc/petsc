IPETSCDIR = ../../../..

ALL: lib

CFLAGS   =  $(PETSC_INCLUDE) -I../..  \
          $(CONF) -D__DIR__='"snes/impls/umls/"'
SOURCEC  = umls.c search.c
SOURCEF  = 
SOURCEH  = umls.h
OBJSC    = umls.o search.o
OBJSF    = 
LIBBASE  = libpetscsnes
MANSEC   = n

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

