ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../.. -I$(ITOOLSDIR)/pinclude\
           $(CONF) -D__DIR__='"pc/impls/sor/"'
SOURCEC  = sor.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = sor.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
DIRS     = 
MANSEC   = 4

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
