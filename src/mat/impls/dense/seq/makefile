ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../..  \
           -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = dense.c 
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = dense.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
MANSEC   = 2

include $(ITOOLSDIR)/bmake/$(PARCH)
