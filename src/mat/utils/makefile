ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I.. -I$(ITOOLSDIR)/pinclude\
           $(CONF) -D__DIR__='"mat/utils"'
SOURCEC  = gcreate.c 
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = gcreate.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
DIRS     = complex
MANSEC   = 2

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
