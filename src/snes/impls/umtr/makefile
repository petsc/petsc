IPETSCDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../..  \
          -I$(IPETSCDIR)/pinclude $(CONF) -I$(IPETSCDIR)/src/ksp \
          -D__DIR__='"snes/impls/trm/"'
SOURCEC  = umtr.c qcg.c
SOURCEF  =
SOURCEH  = umtr.h qcg.h
OBJSC    = umtr.o qcg.o
OBJSF    =
WSOURCEC = 
WOBJS    = 
LIBBASE  = libpetscsnes
LINCLUDE = $(SOURCEH)
MANSEC   = n

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)

