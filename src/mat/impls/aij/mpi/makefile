IPETSCDIR = ../../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I../seq \
           -I$(IPETSCDIR)/pinclude $(CONF)
SOURCEC  = mpiaij.c mmaij.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = mpiaij.o mmaij.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
DIRS     = order
MANSEC   = 2

runexamples:

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)
