ITOOLSDIR = ../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
           -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = precon.c pcset.c pcregis.c
SOURCEF  =
SOURCEH  = 
OBJSC    = precon.o pcset.o pcregis.o
OBJSF    =
WSOURCEC = preconf.c
WOBJS    = preconf.o
LIBBASE  = libpetscpc
LINCLUDE = $(SOURCEH)
MANSEC   = 4

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
