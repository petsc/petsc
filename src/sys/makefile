ITOOLSDIR = ../../

ALL: lib 
CFLAGS    = $(OPT) $(INCLUDEDIRS) $(CONF) -D__DIR__='"sys/"' \
            -I$(ITOOLSDIR)/pinclude
SOURCEC   = tr.c arch.c txt.c file.c  nreg.c try.c comsort.c destroy.c
SOURCEF   =
SOURCEH   =
OBJSC     = tr.o arch.o txt.o file.o  nreg.o try.o comsort.o destroy.o
OBJSF     =
WSOURCEC  = archf.c   
WOBJS     = archf.o  
MANSEC    = 7

LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = error

runexamples:

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
