ITOOLSDIR = ..

ALL: lib workers
CFLAGS    = $(OPT) -I$(ITOOLSDIR)/include -I$(ITOOLSDIR)
SOURCEC   = error.c pprint.c getopts.c nreg.c
SOURCEF  =
WSOURCEC   =
SOURCEH  =
OBJSC      = error.o pprint.o getopts.o nreg.o
WOBJS      =
OBJSF =
LOCAL_INCLUDE = $(SOURCEH)
LIBBASE = libpetscsystem
DIRS = time

example_progs: 
run_examples:
test_progs:
run_tests:

# This is because I got sick of different examples and 
# changing makefiles for parallel and sequential code
workers: fmain.c cmain.c
	$(CC) -c $(CFLAGS) $(BASEOPT) fmain.c cmain.c
	mv fmain.o $(LDIR)/fmain.o
	mv cmain.o $(LDIR)/cmain.o

example: example.c lib
	$(CC) $(CFLAGS) $(BASEOPT) -o example example.c $(LIBNAME)
	rm example.o

tsort: tsort.c 
	$(CC) $(CFLAGS) $(BASEOPT) -o tsort tsort.c $(LIBNAME)

maltest: maltest.c 
	$(CC) $(CFLAGS) $(BASEOPT) -o maltest maltest.c $(LIBNAME)

sbtest: sbtest.c 
	$(CC) $(CFLAGS) $(BASEOPT) -o sbtest sbtest.c $(LIBNAME)
	
farg: farg.f
	$(FC) $(CFLAGS) $(BASEOPT) -o farg farg.f $(FLIB) $(LIBNAME)

# IRIX requires -lfpe
sn: sn.c lib
	$(CC) $(CFLAGS) $(BASEOPT) -o sn sn.c $(LIBNAME) $(LDIR)tools.a -lm 
sn860: sn.c fsr860.s lib
	$(CC) $(CFLAGS) $(BASEOPT) -o sn860 sn.c fsr860.s $(LIBNAME) $(LDIR)tools.a -lm


	
