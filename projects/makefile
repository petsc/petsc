# $Id: makefile,v 1.58 2001/08/22 17:58:52 balay Exp $ 

all: copyexamples unix2dos_projects

copyexamples: 
	cp -f ../src/sles/examples/tutorials/ex2.c c/sles/ex2/ex2.c
	cp -f ../src/sles/examples/tutorials/ex2.c c++/sles/ex2/ex2.cpp
	cp -f ../src/sles/examples/tutorials/ex2.c complex/sles/ex2/ex2.cpp
	cp -f ../src/sles/examples/tutorials/ex2f.F fortran/sles/ex2f/ex2f.F

unixtodos_projects:
	unix2dos c/sles/ex2/ex2.dsp         c/sles/ex2/ex2.dsw
	unix2dos c++/sles/ex2/ex2.dsp       c++/sles/ex2/ex2.dsw
	unix2dos complex/sles/ex2/ex2.dsp   complex/sles/ex2/ex2.dsw
	unix2dos fortran/sles/ex2f/ex2f.dsp fortran/sles/ex2f/ex2f.dsw