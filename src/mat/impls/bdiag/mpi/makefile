IPETSCDIR = ../../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I../seq \
           -I$(IPETSCDIR)/pinclude $(CONF)
SOURCEC  = mmbdiag.c mpibdiag.c
SOURCEF  =
WSOURCE = 
SOURCEH  = mpibdiag.h
OBJSC    = mmbdiag.o mpibdiag.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
