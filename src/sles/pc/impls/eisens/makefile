IPETSCDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../.. -I$(IPETSCDIR)/pinclude\
           $(CONF) -D__DIR__='"pc/impls/eisen/"'
SOURCEC  = eisen.c
SOURCEF  =
WSOURCEC = 
SOURCEH  = 
OBJSC    = eisen.o
WOBJS    = 
OBJSF    =
LIBBASE  = libpetscsles
LINCLUDE = $(SOURCEH)
DIRS     = 
MANSEC   = 4

runexamples:

include $(IPETSCDIR)/bmake/$(PARCH)/$(PARCH)
