#requirespackage  'PETSC_HAVE_MOAB'

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/moab/examples/tests/
EXAMPLESC       = ex1.cxx ex2.cxx ex3.cxx ex4.cxx
EXAMPLESF       =
MANSEC          = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
	${RM} -f ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_DM_LIB}
	${RM} -f ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_DM_LIB}
	${RM} -f ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_DM_LIB}
	${RM} -f ex4.o


#--------------------------------------------------------------------------
runex1:
	-@${MPIEXEC} -n 1 ./ex1 > ex1_1.tmp 2>&1;\
	   if (${DIFF} output/ex1_1.out ex1_1.tmp) then true ;  \
	   else printf "${PWD}\nPossible problem with runex1, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex1_1.tmp

runex2:
	-@${MPIEXEC} -n 1 ./ex2 -debug -fields v1,v2,v3 > ex2_1.tmp 2>&1;\
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with runex2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex2_1.tmp

runex3_1_1:
	-@${MPIEXEC} -n 1 ./ex3 -debug -n 2 -dim 2 -levels 2 -simplex > ex3_1_1.tmp 2>&1;\
	   if (${DIFF} output/ex3_1_1.out ex3_1_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with runex3, diffs above \n========================================="; fi ;\
	   ${RM} -f ex3_1_1.tmp

runex3_1_2:
	-@${MPIEXEC} -n 1 ./ex3 -debug -n 2 -dim 3 -levels 2 > ex3_1_2.tmp 2>&1;\
	   if (${DIFF} output/ex3_1_2.out ex3_1_2.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with runex3, diffs above \n========================================="; fi ;\
	   ${RM} -f ex3_1_2.tmp

runex4:
	-@${MPIEXEC} -n 1 ./ex4 -debug -file input/ex4_in.h5m -dim 2 -levels 2 > ex4_1_1.tmp 2>&1;\
	   if (${DIFF} output/ex4_1_1.out ex4_1_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with runex4, diffs above \n========================================="; fi ;\
	   ${RM} -f ex4_1_1.tmp

TESTEXAMPLES_MOAB = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm ex3.PETSc runex3_1_1 runex3_1_2 ex3.rm ex4.PETSc runex4 ex4.rm ex5.PETSc runex5 ex5.rm

include ${PETSC_DIR}/lib/petsc/conf/test
