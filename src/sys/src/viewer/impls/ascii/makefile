ITOOLSDIR = ../../../../


ALL: lib 
CFLAGS    = $(OPT) -I../.. $(INCLUDEDIRS) -I$(ITOOLSDIR)/pinclude \
            $(XINCLUDEDIR) $(CONF) -D__DIR__='"viewer/impls/file/"'
SOURCEC   = filev.c
SOURCEF   =
SOURCEH   =
OBJSC     = filev.o
OBJSF     =
WSOURCEC  =   
WOBJS     = 
MANSEC    = 7
LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = 

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
