ALL: lib 
ITOOLSDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(ITOOLSDIR)/pinclude  -I../../ -D__DIR__='"ksp/impls/lsqr/"'
SOURCEC  = lsqr.c
SOURCEF  =
WSOURCEC = 
OBJSC    = lsqr.o
OBJSF    =
WOBJS    = 
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
