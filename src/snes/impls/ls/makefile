ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../..  \
          -I$(ITOOLSDIR)/pinclude $(CONF) -D__DIR__='"snes/impls/ls/"'
SOURCEC  = ls.c
SOURCEF  =
SOURCEH  = 
OBJSC    = ls.o
OBJSF    =
WSOURCEC = 
WOBJS    = 
LIBBASE  = libpetscsnes
LINCLUDE = $(SOURCEH)
MANSEC   = n

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

runexamples:

