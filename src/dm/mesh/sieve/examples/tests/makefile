
CFLAGS	        = -I${PETSC_DIR}/src/dm/mesh/sieve
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        = 
LOCDIR          = src/dm/mesh/sieve/examples/tests/
EXAMPLESC       = test1.c, test2.c
EXAMPLESF       = 
MANSEC          = DA
NP              = 1
DISPLAY         = :0.0
EXTRA_ARGS      =

include ${PETSC_DIR}/bmake/common/base

test1: test1.o   chkopts
	-${CLINKER} -o test1 test1.o  ${PETSC_DM_LIB}
	${RM} -f test1.o

test2: test2.o   chkopts
	-${CLINKER} -o test2 test2.o  ${PETSC_DM_LIB}
	${RM} -f test2.o


#-----------------------------------------------------------------
runtest1:
	-@${MPIRUN} -np ${NP} ./test1 ${EXTRA_ARGS}

debugtest1:
	-@${MPIRUN} -np ${NP} ./test1 -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valtest1:
	-@valgrind --tool=memcheck ./test1 ${EXTRA_ARGS}

runtest1_2:
	-@${MPIRUN} -np 2 ./test1 ${EXTRA_ARGS}

runtest1_3:
	-@${MPIRUN} -np 3 ./test1 ${EXTRA_ARGS}
#-----------------------------------------------------------------
runtest2:
	-@${MPIRUN} -np ${NP} ./test2 ${EXTRA_ARGS}

debugtest2:
	-@${MPIRUN} -np ${NP} ./test2 -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valtest2:
	-@valgrind --tool=memcheck ./test2 ${EXTRA_ARGS}


TESTEXAMPLES_C		  = test1.PETSc runtest1 test1.rm
TESTEXAMPLES_C_X11	  = 
TESTEXAMPLES_FORTRAN	  = 
TESTEXAMPLES_F90          = 
TESTEXAMPLES_C_X11_MPIUNI = 
TESTEXAMPLES_C_COMPLEX	  = 
TESTEXAMPLES_C_NOCOMPLEX  = 
TESTEXAMPLES_13		  = 


testclean: clean
	${RM} test1 test2

include ${PETSC_DIR}/bmake/common/test
