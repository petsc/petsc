IPETSCDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I../.. \
           -I$(IPETSCDIR)/pinclude -I$(IPETSCDIR)/src/is $(CONF)
SOURCEC  = dvec3.c bvec2.c 
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = dvec3.o bvec2.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 1

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)

runexamples:
