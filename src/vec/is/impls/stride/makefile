IPETSCDIR = ../../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../../.. -I$(IPETSCDIR) \
           -I$(IPETSCDIR)/pinclude $(CONF) -D__DIR__='"is/impls/seq/stride/"'
SOURCEC  = stride.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = stride.o 
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscvec
LINCLUDE = $(SOURCEH)
MANSEC   = 6

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)

runexamples:
