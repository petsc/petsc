ALL: lib 
IPETSCDIR = ../../../../

CFLAGS   = $(OPT) $(CONF) $(INCLUDEDIRS) -I../..  \
           -I$(IPETSCDIR)/pinclude -D__DIR__='"ksp/impls/rich/"'
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

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)
