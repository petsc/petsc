ITOOLSDIR = ../../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I../seq \
           -I$(ITOOLSDIR)/pinclude $(BS_INCLUDE) $(CONF)
SOURCEC  = mpirowbs.c bsinit.c cholbs.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = mpirowbs.o bsinit.o cholbs.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
DIRS     =
MANSEC   = 2

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
