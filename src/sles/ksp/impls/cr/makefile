ALL: lib 

ITOOLSDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(ITOOLSDIR)/pinclude -I../.. -D__DIR__='"ksp/impls/cr/"'
FFLAGS   =
SOURCEC  = cr.c 
SOURCEF  = 
WSOURCEC = 
OBJSC    = cr.o 
WOBJS    =  
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
