ITOOLSDIR = ../../

ALL: lib workers
CFLAGS    = $(OPT) $(INCLUDEDIRS) $(CONF)
SOURCEC   = tr.c arch.c txt.c file.c  nreg.c
SOURCEF   =
SOURCEH   =
OBJSC     = tr.o arch.o txt.o file.o  nreg.o
OBJSF     =
WSOURCEC  = archf.c   
WOBJS     = archf.o  
MANSEC    = 7

LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = error

# This is because I got sick of different examples and 
# changing makefiles for parallel and sequential code
workers: fmain.c cmain.c
	$(CC) -c $(CFLAGS) $(BASEOPT) fmain.c cmain.c
	mv fmain.o $(LDIR)/fmain.o
	mv cmain.o $(LDIR)/cmain.o

include $(ITOOLSDIR)/bmake/$(PARCH)/$(PARCH)
