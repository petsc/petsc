CFLAGS	        = -I${PETSC_DIR}/src/dm/mesh/sieve
FFLAGS	        =
CPPFLAGS        = -DALE_USE_LOGGING -DALE_HAVE_CXX_ABI# -DALE_USE_DEBUGGING
FPPFLAGS        = 
LOCDIR          = src/dm/mesh/examples/tutorials/
EXAMPLESC       = ex0.c ex1.c
EXAMPLESF       = 
MANSEC          = DA
NP              = 1
DISPLAY         = :0.0
EXTRA_ARGS      =

include ${PETSC_DIR}/bmake/common/base

ex0: ex0.o chkopts
	-${CLINKER} -o ex0 ex0.o ${PETSC_DM_LIB}
	${RM} -f ex0.o

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

#-----------------------------------------------------------------
runex0:
	-@${MPIRUN} -np ${NP} ./ex0 -log_summary ${EXTRA_ARGS}

debugex0:
	-@${MPIRUN} -np ${NP} ./ex0 -log_summary -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valex0:
	-@valgrind --tool=memcheck ./ex0 ${EXTRA_ARGS}

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
