IPETSCDIR = ../../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I../seq \
           -I$(IPETSCDIR)/pinclude $(CONF)
SOURCEC  = mpiaij.c mmaij.c maijfact.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = mpiaij.o mmaij.o maijfact.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
DIRS     = order
MANSEC   = 2

include $(IPETSCDIR)/bmake/$(PETSC_ARCH)/$(PETSC_ARCH)
