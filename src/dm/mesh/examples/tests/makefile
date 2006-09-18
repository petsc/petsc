CPPFLAGS        = -DALE_HAVE_CXX_ABI -DALE_USE_LOGGING -DALE_USE_DEBUGGING -DALE_LOGGING_USE_EVENTS -DALE_LOGGING_LOG_MEM -DALE_LOGGING_VERBOSE #-DALE_LOGGING_USE_STAGES
CFLAGS	        = -I${PETSC_DIR}/src/dm/mesh/sieve ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE}
FFLAGS	        =
FPPFLAGS        = 
LOCDIR          = src/dm/mesh/examples/tutorials/
EXAMPLESC       = ex00.c ex01.c ex02.c ex03.c ex1.c
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

ex03: ex03.o chkopts
	-${CLINKER} -o ex03 ex03.o ${PETSC_DM_LIB}
	${RM} -f ex03.o

ex04: ex04.o chkopts
	-${CLINKER} -o ex04 ex04.o ${PETSC_DM_LIB}
	${RM} -f ex04.o

ex05: ex05.o chkopts
	-${CLINKER} -o ex05 ex05.o ${PETSC_DM_LIB}
	${RM} -f ex05.o

ex06: ex06.o chkopts
	-${CLINKER} -o ex06 ex06.o ${PETSC_DM_LIB}
	${RM} -f ex06.o

ex07: ex07.o chkopts
	-${CLINKER} -o ex07 ex07.o ${PETSC_DM_LIB}
	${RM} -f ex07.o

ex08: ex08.o chkopts
	-${CLINKER} -o ex08 ex08.o ${PETSC_DM_LIB}
	${RM} -f ex08.o

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

sifter1: sifter1.o chkopts
	-${CLINKER} -o sifter1 sifter1.o ${PETSC_DM_LIB}
	${RM} -f sifter1.o

sieve2: sieve2.o chkopts
	-${CLINKER} -o sieve2 sieve2.o ${PETSC_DM_LIB}
	${RM} -f sieve2.o

topology1: topology1.o chkopts
	-${CLINKER} -o topology1 topology1.o ${PETSC_DM_LIB}
	${RM} -f topology1.o

section1: section1.o chkopts
	-${CLINKER} -o section1 section1.o ${PETSC_DM_LIB}
	${RM} -f section1.o

mesh1: mesh1.o chkopts
	-${CLINKER} -o mesh1 mesh1.o ${PETSC_DM_LIB}
	${RM} -f mesh1.o

overlap1: overlap1.o chkopts
	-${CLINKER} -o overlap1 overlap1.o ${PETSC_DM_LIB}
	${RM} -f overlap1.o

overlap2: overlap2.o chkopts
	-${CLINKER} -o overlap2 overlap2.o ${PETSC_DM_LIB}
	${RM} -f overlap2.o

xsifter0: xsifter0.o chkopts
	-${CLINKER} -o xsifter0 xsifter0.o ${PETSC_DM_LIB}
	${RM} -f xsifter0.o
runxsifter0:
	-@${MPIRUN} -np ${NP} ./xsifter0 -log_summary ${EXTRA_ARGS}


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
runex03:
	-@${MPIRUN} -np ${NP} ./ex03 ${EXTRA_ARGS}

debugex03:
	-@${MPIRUN} -np ${NP} ./ex03 -log_summary -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valex03:
	-@valgrind --tool=memcheck ./ex03 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex04:
	-@${MPIRUN} -np ${NP} ./ex04 ${EXTRA_ARGS}

debugex04:
	-@${MPIRUN} -np ${NP} ./ex04 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY} 

valex04:
	-@valgrind --tool=memcheck ./ex04 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex05:
	-@${MPIRUN} -np ${NP} ./ex05 ${EXTRA_ARGS}

debugex05:
	-@${MPIRUN} -np ${NP} ./ex05 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY} 

valex05:
	-@valgrind --tool=memcheck ./ex05 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex06:
	-@${MPIRUN} -np ${NP} ./ex06 ${EXTRA_ARGS}

debugex06:
	-@${MPIRUN} -np ${NP} ./ex06 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY} 

valex06:
	-@valgrind --tool=memcheck ./ex06 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex07:
	-@${MPIRUN} -np ${NP} ./ex07 ${EXTRA_ARGS}

debugex07:
	-@${MPIRUN} -np ${NP} ./ex07 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY} 

valex07:
	-@valgrind --tool=memcheck ./ex07 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex08:
	-@${MPIRUN} -np ${NP} ./ex08 ${EXTRA_ARGS}

debugex08:
	-@${MPIRUN} -np ${NP} ./ex08 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY} 

valex08:
	-@valgrind --tool=memcheck ./ex08 ${EXTRA_ARGS}

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
