IPETSCDIR = ../

CFLAGS   = $(OPT) -I$(IPETSCDIR)/include -I.. -I$(IPETSCDIR) $(CONF)
SOURCEC  =
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    =
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
DIRS     = inline is vec ksp sys pc mat sles options draw grid stencil \
           viewer plog snes

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)

runexamples:

