ITOOLSDIR = ../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I$(ITOOLSDIR)/pinclude \
           -I..   $(CONF) -D__DIR__='"mat/src/"'
SOURCEC  = matrix.c 
SOURCEF  =
WSOURCEC = matrixf.c
SOURCEH  = 
OBJSC    = matrix.o 
WOBJS    = matrixf.o
OBJSF    =
LIBBASE  = libpetscmat
LINCLUDE = $(SOURCEH)
MANSEC   = 2

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
