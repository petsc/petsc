ALL: lib 
ITOOLSDIR = ../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS)  \
           -I$(ITOOLSDIR)/pinclude -I../../
SOURCEC  = cheby.c 
SOURCEF  =
WSOURCEC = chebyf.c
OBJSC    = cheby.o 
OBJSF    =
WOBJS    = chebyf.o
LIBBASE  = libpetscksp
LINCLUDE = 

include $(ITOOLSDIR)/bmake/$(PARCH)
