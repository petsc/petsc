ALL: lib 
ITOOLSDIR = ../../../../

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(CONF)  \
           -I$(ITOOLSDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/bcgs/"'
SOURCEC  = bcgs.c
WSOURCEC = 
SOURCEF  =
OBJSC    = bcgs.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
