IPETSCDIR = ../../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I../seq \
           -I$(IPETSCDIR)/pinclude $(CONF)
SOURCEC  = mmbdiag.c mpibdiag.c
SOURCEF  =
SOURCEH  = mpibdiag.h
OBJSC    = mmbdiag.o mpibdiag.o
OBJSF    =
LIBBASE  = libpetscmat
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
