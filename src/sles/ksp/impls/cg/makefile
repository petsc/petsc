ALL: lib 
ITOOLSDIR = ../../../../

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(CONF)  \
           -I$(ITOOLSDIR)/pinclude -I../.. -D__DIR__='"ksp/impls/cg"'
FFLAGS   =
SOURCEC  = cg.c cgeig.c
SOURCEF  = 
WSOURCEC = cgeigf.c
OBJSC    = cg.o cgeig.o 
WOBJS    = cgeigf.o 
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
