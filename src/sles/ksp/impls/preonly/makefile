ALL: lib 
IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude -I../.. -D__DIR__='"ksp/impls/preonly/"'
SOURCEC  = preonly.c
WSOURCEC = 
SOURCEF  =
OBJSC    = preonly.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)
