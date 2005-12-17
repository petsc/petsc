CFLAGS	        = -I${PETSC_DIR}/src/dm/mesh/sieve -I/PETSc3/ArX-2.2.4/src/boost/
FFLAGS	        =
CPPFLAGS        = -DALE_USE_LOGGING -DALE_HAVE_CXX_ABI -DALE_USE_DEBUGGING
FPPFLAGS        = 
LOCDIR          = src/dm/mesh/examples/tutorials/
EXAMPLESC       = ex00.c ex01.c ex02.c ex1.c
EXAMPLESF       = 
MANSEC          = DA
NP              = 1
DISPLAY         = :0.0
EXTRA_ARGS      =

include ${PETSC_DIR}/bmake/common/base

ex00: ex00.o chkopts
	-${CLINKER} -o ex00 ex00.o ${PETSC_DM_LIB}
	${RM} -f ex00.o

ex01: ex01.o chkopts
	-${CLINKER} -o ex01 ex01.o ${PETSC_DM_LIB}
	${RM} -f ex01.o

ex02: ex02.o chkopts
	-${CLINKER} -o ex02 ex02.o ${PETSC_DM_LIB}
	${RM} -f ex02.o

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

#-----------------------------------------------------------------
runex00:
	-@${MPIRUN} -np ${NP} ./ex00 -log_summary ${EXTRA_ARGS}

debugex00:
	-@${MPIRUN} -np ${NP} ./ex00 -log_summary -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valex00:
	-@valgrind --tool=memcheck ./ex00 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex01:
	-@${MPIRUN} -np ${NP} ./ex01 -log_summary ${EXTRA_ARGS}

debugex01:
	-@${MPIRUN} -np ${NP} ./ex01 -log_summary -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valex01:
	-@valgrind --tool=memcheck ./ex01 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex02:
	-@${MPIRUN} -np ${NP} ./ex02 -log_summary ${EXTRA_ARGS}

debugex02:
	-@${MPIRUN} -np ${NP} ./ex02 -log_summary -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valex02:
	-@valgrind --tool=memcheck ./ex02 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex1:
	-@${MPIRUN} -np ${NP} ./ex1 -log_summary ${EXTRA_ARGS}

debugex1:
	-@${MPIRUN} -np ${NP} ./ex1 -log_summary ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY}

valex1:
	-@valgrind --tool=memcheck ./ex1 ${EXTRA_ARGS}

TESTEXAMPLES_C            = 
TESTEXAMPLES_C_X11        = 
TESTEXAMPLES_FORTRAN      = 
TESTEXAMPLES_F90          = 
TESTEXAMPLES_C_X11_MPIUNI = 
TESTEXAMPLES_C_COMPLEX    = 
TESTEXAMPLES_C_NOCOMPLEX  = 
TESTEXAMPLES_13           = 

include ${PETSC_DIR}/bmake/common/test
