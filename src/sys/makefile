ITOOLSDIR = ..

ALL: lib workers
CFLAGS    = $(OPT) -I$(ITOOLSDIR)/include -I$(ITOOLSDIR) $(CONF)
SOURCEC   = tr.c arch.c txt.c file.c error.c  getopts.c nreg.c
SOURCEF   =
SOURCEH   =
OBJSC     = tr.o arch.o txt.o file.o error.o  getopts.o nreg.o
OBJSF     =
WSOURCEC  = archf.c errorf.c  
WOBJS     = archf.o errorf.o 
MANSEC    = 7

LINCLUDE  = $(SOURCEH)
LIBBASE   = libpetscsys
DIRS      = time

# This is because I got sick of different examples and 
# changing makefiles for parallel and sequential code
workers: fmain.c cmain.c
	$(CC) -c $(CFLAGS) $(BASEOPT) fmain.c cmain.c
	mv fmain.o $(LDIR)/fmain.o
	mv cmain.o $(LDIR)/cmain.o

include $(ITOOLSDIR)/bmake/$(PARCH)
