ALL: clean setup
CFLAGS          =
FFLAGS          =
CXXFLAGS        = -std=c++11
CPPFLAGS        =
FPPFLAGS        =
CLEANFILES      = *.tap *.txt *.o
CC              = g++

INCLUDES        =
LIB             = ${PETSC_TS_LIB} -L${ADOLC_BUILDDIR} -ladolc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# TODO: See https://randu.org/tutorials/c/libraries.php

%.o: %.cpp
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)

setup: allocation.o matfree.o sparse.o tests.o
	ar rc libpetscadolc.a $^
	ranlib libpetscadolc.a

include ${PETSC_DIR}/lib/petsc/conf/test
