ALL: lib 
ITOOLSDIR = ../../../

CFLAGS   = $(OPT) $(INCLUDEDIRS) $(CONF)  \
           -I$(ITOOLSDIR)/pinclude -I../..
FFLAGS   =
SOURCEC  = cg.c cgeig.c
SOURCEF  = 
WSOURCEC = cgeigf.c
OBJSC    = cg.o cgeig.o 
WOBJS    = cgeigf.o 
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 

include $(ITOOLSDIR)/bmake/$(PARCH)
