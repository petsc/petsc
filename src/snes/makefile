IPETSCDIR = ../../

ALL: lib

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = snes.h
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscsnes
LINCLUDE = $(SOURCEH)
DIRS     = src impls  examples

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)

runexamples:
