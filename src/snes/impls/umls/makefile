IPETSCDIR = ../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../..  \
          -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"snes/impls/umls/"'
SOURCEC  = umls.c
SOURCEF  = mcsrch.f
SOURCEH  = umls.h
OBJSC    = umls.o
OBJSF    = mcsrch.o
WSOURCEC = 
WOBJS    = 
LIBBASE  = libpetscsnes
MANSEC   = n

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

