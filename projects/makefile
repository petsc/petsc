# $Id: makefile,v 1.58 2001/08/22 17:58:52 balay Exp $ 

all: copyexamples unix2dos_projects

copyexamples: 
	cp -f ../src/ksp/ksp/examples/tutorials/ex2.c c/ksp/ex2/ex2.c
	cp -f ../src/ksp/ksp/examples/tutorials/ex2.c c++/ksp/ex2/ex2.cpp
	cp -f ../src/ksp/ksp/examples/tutorials/ex2.c complex/ksp/ex2/ex2.cpp
	cp -f ../src/ksp/ksp/examples/tutorials/ex2f.F fortran/ksp/ex2f/ex2f.F

unix2dos_projects:
	unix2dos c/ksp/ex2/ex2.dsp         c/ksp/ex2/ex2.dsw
	unix2dos c++/ksp/ex2/ex2.dsp       c++/ksp/ex2/ex2.dsw
	unix2dos complex/ksp/ex2/ex2.dsp   complex/ksp/ex2/ex2.dsw
	unix2dos fortran/ksp/ex2f/ex2f.dsp fortran/ksp/ex2f/ex2f.dsw