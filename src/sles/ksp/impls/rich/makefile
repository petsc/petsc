ALL: lib 
ITOOLSDIR = ../../..

CFLAGS   = $(OPT) $(CONF) -I$(ITOOLSDIR) -I../.. -I$(ITOOLSDIR)/include \
           -I$(ITOOLSDIR)/pinclude
SOURCEC  = rich.c
WSOURCEC = richf.c
SOURCEF  =
OBJSC    = rich.o 
WOBJS    = richf.o 
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

include $(ITOOLSDIR)/bmake/$(PARCH)

