ALL: lib 
ITOOLSDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(ITOOLSDIR)/pinclude -I../.. -D__DIR__='"ksp/impls/preonly/"'
SOURCEC  = preonly.c
WSOURCEC = 
SOURCEF  =
OBJSC    = preonly.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
