ITOOLSDIR = ../../../

ALL: lib 
CFLAGS    = $(OPT) $(INCLUDEDIRS) -I$(ITOOLSDIR)/pinclude $(CONF)
SOURCEC   = adebug.c  error.c signal.c
SOURCEF   =
SOURCEH   =
OBJSC     = adebug.o  error.o signal.o
OBJSF     =
WSOURCEC  = errorf.c  
WOBJS     = errorf.o 
MANSEC    = 7

LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = 

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
