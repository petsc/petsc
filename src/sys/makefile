ITOOLSDIR = ../../

ALL: lib 
CFLAGS    = $(OPT) $(INCLUDEDIRS) $(CONF)
SOURCEC   = tr.c arch.c txt.c file.c  nreg.c try.c
SOURCEF   =
SOURCEH   =
OBJSC     = tr.o arch.o txt.o file.o  nreg.o try.o
OBJSF     =
WSOURCEC  = archf.c   
WOBJS     = archf.o  
MANSEC    = 7

LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = error

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
