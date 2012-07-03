CPPFLAGS        =
CFLAGS	        =
FFLAGS	        =
FPPFLAGS        = 
LOCDIR          = src/dm/mesh/examples/tests/
EXAMPLESC       = ex00.c ex01.c ex02.c ex03.c ex1.c
EXAMPLESF       = 
MANSEC          = DM
NP              = 1
MATTSDISPLAY    = :0.0
EXTRA_ARGS      = 

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex3: ex3.o chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_DM_LIB}
	${RM} -f ex3.o

ex4: ex4.o chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_DM_LIB}
	${RM} -f ex4.o

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

meshing: meshing.o chkopts
	-${CLINKER} -o meshing meshing.o ${PETSC_DM_LIB}
	${RM} -f meshing.o

overlap1: overlap1.o chkopts
	-${CLINKER} -o overlap1 overlap1.o ${PETSC_DM_LIB}
	${RM} -f overlap1.o

overlap2: overlap2.o chkopts
	-${CLINKER} -o overlap2 overlap2.o ${PETSC_DM_LIB}
	${RM} -f overlap2.o

conversion1: conversion1.o chkopts
	-${CLINKER} -o conversion1 conversion1.o ${PETSC_DM_LIB}
	${RM} -f conversion1.o

operator1: operator1.o chkopts
	-${CLINKER} -o operator1 operator1.o ${PETSC_DM_LIB}
	${RM} -f operator1.o

subset: subset.o chkopts
	-${CLINKER} -o subset subset.o ${PETSC_DM_LIB}
	${RM} -f subset.o

unitTests: stl1.o sifter1.o sieve1.o isieve1.o section1.o isection1.o imesh1.o distribution1.o idistribution1.o unitTests.o chkopts
	-${CLINKER} -o unitTests stl1.o sifter1.o sieve1.o isieve1.o section1.o isection1.o imesh1.o distribution1.o idistribution1.o unitTests.o ${PETSC_DM_LIB} -lcppunit
	#${RM} -f stl1.o sifter1.o sieve1.o isieve1.o section1.o isection1.o imesh1.o distribution1.o idistribution1.o unitTests.o

memoryTests: memoryTests.o chkopts
	-${CLINKER} -o memoryTests memoryTests.o ${PETSC_DM_LIB}
	${RM} -f memoryTests.o

sieveScalability: sieveScalability.o chkopts
	-${CLINKER} -o sieveScalability sieveScalability.o ${PETSC_DM_LIB}
	${RM} -f sieveScalability.o

distributionTests: distributionTests.o chkopts
	-${CLINKER} -o distributionTests distributionTests.o ${PETSC_DM_LIB}
	${RM} -f distributionTests.o

refinementTest: refinementTest.o chkopts
	-${CLINKER} -o refinementTest refinementTest.o ${PETSC_DM_LIB}
	${DSYMUTIL} refinementTest
	${RM} -f refinementTest.o

sieveTests: sieveTests.o chkopts
	-${CLINKER} -o sieveTests sieveTests.o ${PETSC_DM_LIB}
	${RM} -f sieveTests.o

rbfPrecon: rbfPrecon.o chkopts
	-${CLINKER} -o rbfPrecon rbfPrecon.o ${PETSC_SNES_LIB}
	${RM} -f rbfPrecon.o

TestMeshExodus: TestMeshExodus.o chkopts
	-${CLINKER} -o TestMeshExodus TestMeshExodus.o ${PETSC_SNES_LIB}
	${RM} -f TestMeshExodus.o

TestMesh: TestMesh.o chkopts
	-${CLINKER} -o TestMesh TestMesh.o ${PETSC_SNES_LIB}
	${DSYMUTIL} TestMesh
	${RM} -f TestMesh.o
#-----------------------------------------------------------------
BOOST_LIBS = -Wl,-rpath,${BOOST_DIR}/lib -L${BOOST_DIR}/lib -lboost_program_options-gcc41
args0: args0.o chkopts
	-${CLINKER} -o args0 args0.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f args0.o

ppargs0: chkopts
	-${CXCPP} args0.cxx

runargs0:
	-@${MPIEXEC} -n ${NP} ./args0 ${EXTRA_ARGS}

debugargs0:
	-@${MPIEXEC} -n ${NP} ./args0 -log_summary -start_in_debugger -display ${MATTSDISPLAY} ${EXTRA_ARGS}

#-----------------------------------------------------------------
xsifter0: xsifter0.o chkopts
	-${CLINKER} -o xsifter0 xsifter0.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsifter0.o
NP = 1
runxsifter0:
	-@${MPIEXEC} -n ${NP} ./xsifter0 ${EXTRA_ARGS}
debugxsifter0:
	-@${MPIEXEC} -n ${NP} ./xsifter0 -log_summary -start_in_debugger -display ${MATTSDISPLAY} ${EXTRA_ARGS}

#-----------------------------------------------------------------
xsifter1: xsifter1.o chkopts
	-${CLINKER} -o xsifter1 xsifter1.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsifter1.o
NP = 1
runxsifter1:
	-@${MPIEXEC} -n ${NP} ./xsifter1 ${EXTRA_ARGS}
debugxsifter1:
	-@${MPIEXEC} -n ${NP} ./xsifter1 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
xsifter2: xsifter2.o chkopts
	-${CLINKER} -o xsifter2 xsifter2.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsifter2.o
NP = 1
runxsifter2:
	-@${MPIEXEC} -n ${NP} ./xsifter2 ${EXTRA_ARGS}
debugxsifter2:
	-@${MPIEXEC} -n ${NP} ./xsifter2 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
xsifter3: xsifter3.o chkopts
	-${CLINKER} -o xsifter3 xsifter3.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsifter3.o
NP = 1
runxsifter3:
	-@${MPIEXEC} -n ${NP} ./xsifter3 ${EXTRA_ARGS}
debugxsifter3:
	-@${MPIEXEC} -n ${NP} ./xsifter3 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
xsieve0: xsieve0.o chkopts
	-${CLINKER} -o xsieve0 xsieve0.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsieve0.o
NP = 1
runxsieve0:
	-@${MPIEXEC} -n ${NP} ./xsieve0 ${EXTRA_ARGS}
debugxsieve0:
	-@${MPIEXEC} -n ${NP} ./xsieve0 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
xsieve1: xsieve1.o chkopts
	-${CLINKER} -o xsieve1 xsieve1.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsieve1.o
NP = 1
runxsieve1:
	-@${MPIEXEC} -n ${NP} ./xsieve1 ${EXTRA_ARGS}
debugxsieve1:
	-@${MPIEXEC} -n ${NP} ./xsieve1 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
xsieve2: xsieve2.o chkopts
	-${CLINKER} -o xsieve2 xsieve2.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsieve2.o
NP = 1
runxsieve2:
	-@${MPIEXEC} -n ${NP} ./xsieve2 ${EXTRA_ARGS}
debugxsieve2:
	-@${MPIEXEC} -n ${NP} ./xsieve2 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
xsieve3: xsieve3.o chkopts
	-${CLINKER} -o xsieve3 xsieve3.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsieve3.o
NP = 1
runxsieve3:
	-@${MPIEXEC} -n ${NP} ./xsieve3 ${EXTRA_ARGS}
debugxsieve3:
	-@${MPIEXEC} -n ${NP} ./xsieve3 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
xsieve4: xsieve4.o chkopts
	-${CLINKER} -o xsieve4 xsieve4.o ${PETSC_DM_LIB} ${BOOST_LIBS}
	${RM} -f xsieve4.o
NP = 1
runxsieve4:
	-@${MPIEXEC} -n ${NP} ./xsieve4 ${EXTRA_ARGS}
debugxsieve4:
	-@${MPIEXEC} -n ${NP} ./xsieve4 ${EXTRA_ARGS} -log_summary -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
runex00:
	-@${MPIEXEC} -n ${NP} ./ex00 -log_summary ${EXTRA_ARGS}

debugex00:
	-@${MPIEXEC} -n ${NP} ./ex00 -log_summary -start_in_debugger -display ${MATTSDISPLAY} ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex01:
	-@${MPIEXEC} -n ${NP} ./ex01 -log_summary ${EXTRA_ARGS}

debugex01:
	-@${MPIEXEC} -n ${NP} ./ex01 -log_summary -start_in_debugger -display ${MATTSDISPLAY} ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex02:
	-@${MPIEXEC} -n ${NP} ./ex02 -log_summary ${EXTRA_ARGS}

debugex02:
	-@${MPIEXEC} -n ${NP} ./ex02 -log_summary -start_in_debugger -display ${MATTSDISPLAY} ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex03:
	-@${MPIEXEC} -n ${NP} ./ex03 ${EXTRA_ARGS}

debugex03:
	-@${MPIEXEC} -n ${NP} ./ex03 -log_summary -start_in_debugger -display ${MATTSDISPLAY} ${EXTRA_ARGS}

#-----------------------------------------------------------------
runex04:
	-@${MPIEXEC} -n ${NP} ./ex04 ${EXTRA_ARGS}

debugex04:
	-@${MPIEXEC} -n ${NP} ./ex04 ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
runex05:
	-@${MPIEXEC} -n ${NP} ./ex05 ${EXTRA_ARGS}

debugex05:
	-@${MPIEXEC} -n ${NP} ./ex05 ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
runex06:
	-@${MPIEXEC} -n ${NP} ./ex06 ${EXTRA_ARGS}

debugex06:
	-@${MPIEXEC} -n ${NP} ./ex06 ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
runex07:
	-@${MPIEXEC} -n ${NP} ./ex07 ${EXTRA_ARGS}

debugex07:
	-@${MPIEXEC} -n ${NP} ./ex07 ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
runex08:
	-@${MPIEXEC} -n ${NP} ./ex08 ${EXTRA_ARGS}

debugex08:
	-@${MPIEXEC} -n ${NP} ./ex08 ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY} 

#-----------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n ${NP} ./ex1 -log_summary ${EXTRA_ARGS}

debugex1:
	-@${MPIEXEC} -n ${NP} ./ex1 -log_summary ${EXTRA_ARGS} -start_in_debugger -display ${MATTSDISPLAY}

TESTEXAMPLES_C            = 
TESTEXAMPLES_C_X        = 
TESTEXAMPLES_FORTRAN      = 
TESTEXAMPLES_F90          = 
TESTEXAMPLES_C_X_MPIUNI = 
TESTEXAMPLES_C_COMPLEX    = 
TESTEXAMPLES_C_NOCOMPLEX  = 
TESTEXAMPLES_13           = 

include ${PETSC_DIR}/conf/test
