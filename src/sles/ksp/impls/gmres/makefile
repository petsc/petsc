ALL: lib

ITOOLSDIR = ../../../

CFLAGS   = $(OPT) $(CONF) -I$(ITOOLSDIR) -I$(ITOOLSDIR)/include \
           -I$(ITOOLSDIR)/pinclude -I../../
SOURCEC  = gmres.c borthog.c mmaxpy.c
WSOURCEC = gmresf.c  
SOURCEF  =
OBJSC    = gmres.o borthog.o mmaxpy.o
WOBJS    = gmresf.o 
OBJSF    = 
LIBBASE  = libpetscksp
LINCLUDE = 

include $(ITOOLSDIR)/bmake/$(PARCH)
