ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I$(ITOOLSDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = jacobi none sor direct shell bjacobi mg eisens ilu

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
