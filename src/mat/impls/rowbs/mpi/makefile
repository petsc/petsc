IPETSCDIR = ../../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I../seq \
           -I$(IPETSCDIR)/pinclude $(BS_INCLUDE) $(CONF)
SOURCEC  = mpirowbs.c cholbs.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = mpirowbs.o cholbs.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
DIRS     =
MANSEC   = 2

runexamples:

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)
