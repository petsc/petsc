ALL: lib 
ITOOLSDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(ITOOLSDIR)/pinclude  -I../../ -D__DIR__='"ksp/impls/tcqmr/"'
SOURCEC  = tcqmr.c
SOURCEF  =
WSOURCEC = 
OBJSC    = tcqmr.o
OBJSF    = 
WOBJS    = 
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
