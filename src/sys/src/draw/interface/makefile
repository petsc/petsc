ITOOLSDIR = ../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I..  \
          -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC  = draw.c 
SOURCEF  =
SOURCEH  = 
OBJSC    = draw.o 
OBJSF    =
WSOURCEC = drawf.c
WOBJS    = drawf.o
LIBBASE  = libpetscdraw
LINCLUDE = $(SOURCEH)
MANSEC   = x

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
