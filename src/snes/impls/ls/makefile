ITOOLSDIR = ../../../../

ALL: lib

CFLAGS   = $(OPT) $(INCLUDEDIRS) -I../..  \
          -I$(ITOOLSDIR)/pinclude $(CONF) -D__DIR__='"snes/impls/newls1/"'
SOURCEC  = newls1.c
SOURCEF  =
SOURCEH  = 
OBJSC    = newls1.o
OBJSF    =
WSOURCEC = 
WOBJS    = 
LIBBASE  = libpetscsnes
LINCLUDE = $(SOURCEH)
MANSEC   = n

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)

runexamples:

