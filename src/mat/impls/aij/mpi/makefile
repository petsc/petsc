ITOOLSDIR = ../../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I../seq \
           -I$(ITOOLSDIR)/pinclude $(CONF)
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

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
