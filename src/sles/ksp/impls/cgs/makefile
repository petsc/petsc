ALL: lib 
IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/cgs/"'
SOURCEC  = cgs.c
WSOURCEC = 
SOURCEF  =
OBJSC    = cgs.o
WOBJS    = 
OBJSF    = 
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)
