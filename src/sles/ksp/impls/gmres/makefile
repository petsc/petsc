ALL: lib

ITOOLSDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) \
           -I$(ITOOLSDIR)/pinclude -I../../ -D__DIR__='"ksp/impls/gmres/"'
SOURCEC  = gmres.c borthog.c mmaxpy.c
WSOURCEC = gmresf.c  
SOURCEF  =
OBJSC    = gmres.o borthog.o mmaxpy.o
WOBJS    = gmresf.o 
OBJSF    = 
LIBBASE  = libpetscksp
LINCLUDE = 
MANSEC   = 5

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
