IPETSCDIR = ../../

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscpc
LINCLUDE = $(SOURCEH)
DIRS     = utils src impls  examples

runexamples:

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)
