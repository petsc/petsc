IPETSCDIR = ../../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I../seq \
           -I$(IPETSCDIR)/pinclude $(CONF)
SOURCEC  = mmbdiag.c mpibdiag.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = mpibdiag.h
OBJSC    = mmbdiag.o mpibdiag.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
MANSEC   = 2

runexamples:

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
