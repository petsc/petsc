ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I$(ITOOLSDIR)/pinclude \
           -I../.. -I$(ITOOLSDIR) $(CONF)
SOURCEC  =  
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    =  
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = seq mpi

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

