ALL: lib 
ITOOLSDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(ITOOLSDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/tfqmr/"'
SOURCEC  = tfqmr.c
SOURCEF  =
WSOURCEC = 
OBJSC    = tfqmr.o
OBJSF    =
WOBJS    = 
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
