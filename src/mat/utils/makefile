ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) -I$(ITOOLSDIR)/include -I.. -I$(ITOOLSDIR)/pinclude\
           $(COPT) $(CONF) -D__DIR__='"mat/utils"'
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
