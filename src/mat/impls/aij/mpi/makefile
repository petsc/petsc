IPETSCDIR = ../../../../..

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I../seq \
           -I$(IPETSCDIR)/pinclude $(CONF)
SOURCEC  = mpiaij.c mmaij.c maijfact.c
SOURCEF  =
WSOURCE = 
SOURCEH  = 
OBJSC    = mpiaij.o mmaij.o maijfact.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
DIRS     = order
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
