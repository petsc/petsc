ITOOLSDIR = ../../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I$(ITOOLSDIR) \
           -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = general.c 
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = general.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 6

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

runexamples:
