ALL: lib 
ITOOLSDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) -I../..  \
           -I$(ITOOLSDIR)/pinclude -D__DIR__='"ksp/impls/rich/"'
SOURCEC  = rich.c
WSOURCEC = richf.c
SOURCEF  =
OBJSC    = rich.o 
WOBJS    = richf.o 
OBJSF    =
LIBBASE  = libpetscksp
LINCLUDE = 
MANSEC   = 5

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
