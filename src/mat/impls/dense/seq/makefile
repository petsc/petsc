ITOOLSDIR = ../../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../..  \
           -I$(ITOOLSDIR)/pinclude $(CONF) -D__DIR__='"mat/impls/dense/seq/"'
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

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
