ALL: lib 
IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(IPETSCDIR)/pinclude  -I../../ -D__DIR__='"ksp/impls/lsqr/"'
SOURCEC  = lsqr.c
SOURCEF  =
WSOURCEC = 
OBJSC    = lsqr.o
OBJSF    =
WOBJS    = 
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)
