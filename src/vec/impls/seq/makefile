ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I../.. \
           -I$(ITOOLSDIR)/pinclude -I$(ITOOLSDIR)/src/is $(CONF)
SOURCEC  = dvec3.c bvec2.c 
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = dvec3.o bvec2.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

runexamples:
