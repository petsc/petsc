CPPFLAGS        = ${SIEVE_FLAGS} ${EXTRA_FLAGS}
CFLAGS	        = -I. -I${PETSC_DIR}/src/dm/mesh/sieve ${BOOST_INCLUDE} ${TRIANGLE_INCLUDE} ${TETGEN_INCLUDE}
FFLAGS	        =
FPPFLAGS        = 
LOCDIR          = src/dm/mesh/examples/tutorials/
EXAMPLESC       = ex00.c ex01.c ex02.c ex03.c ex1.c
EXAMPLESF       = 
MANSEC          = DA
NP              = 1
DISPLAY         = :0.0
EXTRA_ARGS      = 

include ${PETSC_DIR}/conf/base

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

mesh2: mesh2.o chkopts
	-${CLINKER} -o mesh2 mesh2.o ${PETSC_DM_LIB}
	${RM} -f mesh2.o

overlap1: overlap1.o chkopts
	-${CLINKER} -o overlap1 overlap1.o ${PETSC_DM_LIB}
	${RM} -f overlap1.o

overlap2: overlap2.o chkopts
	-${CLINKER} -o overlap2 overlap2.o ${PETSC_DM_LIB}
	${RM} -f overlap2.o

operator1: operator1.o chkopts
	-${CLINKER} -o operator1 operator1.o ${PETSC_DM_LIB}
	${RM} -f operator1.o

subset: subset.o chkopts
	-${CLINKER} -o subset subset.o ${PETSC_DM_LIB}
	${RM} -f subset.o

unitTests: sifter1.o sieve1.o section1.o unitTests.o chkopts
	-${CLINKER} -o unitTests sifter1.o sieve1.o section1.o unitTests.o ${PETSC_DM_LIB} -lcppunit
	${RM} -f sifter1.o sieve1.o section1.o unitTests.o

memoryTests: memoryTests.o chkopts
	-${CLINKER} -o memoryTests memoryTests.o ${PETSC_DM_LIB}
	${RM} -f memoryTests.o
#-----------------------------------------------------------------
BOOST_LIBS = -Wl,-rpath,${BOOST_DIR}/lib -L${BOOST_DIR}/lib -lboost_program_options-gcc41
args0: args0.o chkopts
	-${CLINKER} -o args0 args0.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f args0.o

ppargs0: chkopts
	-${CXCPP} args0.cxx

runargs0:
	-@${MPIEXEC} -np ${NP} ./args0 ${EXTRA_ARGS}

debugargs0:
	-@${MPIEXEC} -np ${NP} ./args0 -log_summary -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

#-----------------------------------------------------------------
xsifter0: xsifter0.o chkopts
	-${CLINKER} -o xsifter0 xsifter0.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsifter0.o
NP = 1
runxsifter0:
	-@${MPIEXEC} -np ${NP} ./xsifter0 ${EXTRA_ARGS}
debugxsifter0:
	-@${MPIEXEC} -np ${NP} ./xsifter0 -log_summary -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

#-----------------------------------------------------------------
xsifter1: xsifter1.o chkopts
	-${CLINKER} -o xsifter1 xsifter1.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsifter1.o
NP = 1
runxsifter1:
	-@${MPIEXEC} -np ${NP} ./xsifter1 ${EXTRA_ARGS}
debugxsifter1:
	-@${MPIEXEC} -np ${NP} ./xsifter1 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${DISPLAY} 

valgrindxsifter1:
	-@${MPIEXEC} -np ${NP} valgrind ./xsifter1 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${DISPLAY} 

#-----------------------------------------------------------------
xsifter2: xsifter2.o chkopts
	-${CLINKER} -o xsifter2 xsifter2.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsifter2.o
NP = 1
runxsifter2:
	-@${MPIEXEC} -np ${NP} ./xsifter2 ${EXTRA_ARGS}
debugxsifter2:
	-@${MPIEXEC} -np ${NP} ./xsifter2 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${DISPLAY} 
valgrindxsifter2:
	-@${MPIEXEC} -np ${NP} valgrind ./xsifter2 ${EXTRA_ARGS} -on_error_attach_debugger -display ${DISPLAY} 


#-----------------------------------------------------------------
xsifter3: xsifter3.o chkopts
	-${CLINKER} -o xsifter3 xsifter3.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsifter3.o
NP = 1
runxsifter3:
	-@${MPIEXEC} -np ${NP} ./xsifter3 ${EXTRA_ARGS}
debugxsifter3:
	-@${MPIEXEC} -np ${NP} ./xsifter3 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${DISPLAY} 
valgrindxsifter3:
	-@${MPIEXEC} -np ${NP} valgrind ./xsifter3 ${EXTRA_ARGS} -on_error_attach_debugger -display ${DISPLAY} 

#-----------------------------------------------------------------
xsieve0: xsieve0.o chkopts
	-${CLINKER} -o xsieve0 xsieve0.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsieve0.o
NP = 1
runxsieve0:
	-@${MPIEXEC} -np ${NP} ./xsieve0 ${EXTRA_ARGS}
debugxsieve0:
	-@${MPIEXEC} -np ${NP} ./xsieve0 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${DISPLAY} 
valgrindxsieve0:
	-@${MPIEXEC} -np ${NP} valgrind ./xsieve0 ${EXTRA_ARGS} -on_error_attach_debugger -display ${DISPLAY} 


#-----------------------------------------------------------------
xsieve1: xsieve1.o chkopts
	-${CLINKER} -o xsieve1 xsieve1.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsieve1.o
NP = 1
runxsieve1:
	-@${MPIEXEC} -np ${NP} ./xsieve1 ${EXTRA_ARGS}
debugxsieve1:
	-@${MPIEXEC} -np ${NP} ./xsieve1 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${DISPLAY} 
valgrindxsieve1:
	-@${MPIEXEC} -np ${NP} valgrind ./xsieve1 ${EXTRA_ARGS} -on_error_attach_debugger -display ${DISPLAY} 

#-----------------------------------------------------------------
xsieve2: xsieve2.o chkopts
	-${CLINKER} -o xsieve2 xsieve2.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsieve2.o
NP = 1
runxsieve2:
	-@${MPIEXEC} -np ${NP} ./xsieve2 ${EXTRA_ARGS}
debugxsieve2:
	-@${MPIEXEC} -np ${NP} ./xsieve2 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${DISPLAY} 
valgrindxsieve2:
	-@${MPIEXEC} -np ${NP} valgrind ./xsieve2 ${EXTRA_ARGS} -on_error_attach_debugger -display ${DISPLAY} 

#-----------------------------------------------------------------
xsieve3: xsieve3.o chkopts
	-${CLINKER} -o xsieve3 xsieve3.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsieve3.o
NP = 1
runxsieve3:
	-@${MPIEXEC} -np ${NP} ./xsieve3 ${EXTRA_ARGS}
debugxsieve3:
	-@${MPIEXEC} -np ${NP} ./xsieve3 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${DISPLAY} 
valgrindxsieve3:
	-@${MPIEXEC} -np ${NP} valgrind ./xsieve3 ${EXTRA_ARGS} -on_error_attach_debugger -display ${DISPLAY} 

#-----------------------------------------------------------------
xsieve4: xsieve4.o chkopts
	-${CLINKER} -o xsieve4 xsieve4.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsieve4.o
NP = 1
runxsieve4:
	-@${MPIEXEC} -np ${NP} ./xsieve4 ${EXTRA_ARGS}
debugxsieve4:
	-@${MPIEXEC} -np ${NP} ./xsieve4 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${DISPLAY} 
valgrindxsieve4:
	-@${MPIEXEC} -np ${NP} valgrind ./xsieve4 ${EXTRA_ARGS} -on_error_attach_debugger -display ${DISPLAY} 


#-----------------------------------------------------------------
runex00:
	-@${MPIEXEC} -np ${NP} ./ex00 -log_summary ${EXTRA_ARGS}

debugex00:
	-@${MPIEXEC} -np ${NP} ./ex00 -log_summary -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valex00:
	-@valgrind --tool=memcheck ./ex00 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex01:
	-@${MPIEXEC} -np ${NP} ./ex01 -log_summary ${EXTRA_ARGS}

debugex01:
	-@${MPIEXEC} -np ${NP} ./ex01 -log_summary -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valex01:
	-@valgrind --tool=memcheck ./ex01 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex02:
	-@${MPIEXEC} -np ${NP} ./ex02 -log_summary ${EXTRA_ARGS}

debugex02:
	-@${MPIEXEC} -np ${NP} ./ex02 -log_summary -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valex02:
	-@valgrind --tool=memcheck ./ex02 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex03:
	-@${MPIEXEC} -np ${NP} ./ex03 ${EXTRA_ARGS}

debugex03:
	-@${MPIEXEC} -np ${NP} ./ex03 -log_summary -start_in_debugger -display ${DISPLAY} ${EXTRA_ARGS}

valex03:
	-@valgrind --tool=memcheck ./ex03 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex04:
	-@${MPIEXEC} -np ${NP} ./ex04 ${EXTRA_ARGS}

debugex04:
	-@${MPIEXEC} -np ${NP} ./ex04 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY} 

valex04:
	-@valgrind --tool=memcheck ./ex04 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex05:
	-@${MPIEXEC} -np ${NP} ./ex05 ${EXTRA_ARGS}

debugex05:
	-@${MPIEXEC} -np ${NP} ./ex05 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY} 

valex05:
	-@valgrind --tool=memcheck ./ex05 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex06:
	-@${MPIEXEC} -np ${NP} ./ex06 ${EXTRA_ARGS}

debugex06:
	-@${MPIEXEC} -np ${NP} ./ex06 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY} 

valex06:
	-@valgrind --tool=memcheck ./ex06 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex07:
	-@${MPIEXEC} -np ${NP} ./ex07 ${EXTRA_ARGS}

debugex07:
	-@${MPIEXEC} -np ${NP} ./ex07 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY} 

valex07:
	-@valgrind --tool=memcheck ./ex07 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex08:
	-@${MPIEXEC} -np ${NP} ./ex08 ${EXTRA_ARGS}

debugex08:
	-@${MPIEXEC} -np ${NP} ./ex08 ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY} 

valex08:
	-@valgrind --tool=memcheck ./ex08 ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex1:
	-@${MPIEXEC} -np ${NP} ./ex1 -log_summary ${EXTRA_ARGS}

debugex1:
	-@${MPIEXEC} -np ${NP} ./ex1 -log_summary ${EXTRA_ARGS} -start_in_debugger -display ${DISPLAY}

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

include ${PETSC_DIR}/conf/test
