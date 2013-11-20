
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/moab/examples/tests/
EXAMPLESC       = ex1.cxx ex2.c ex3.c ex4.c
EXAMPLESF       =
MANSEC          = DM

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

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
	   else printf "${PWD}\nPossible problem with with runex1, diffs above\n=========================================\n"; fi ;\
	   ${RM} -f ex1_1.tmp

runex2:
	-@${MPIEXEC} -n 1 ./ex2 -n 10 > ex2_1.tmp 2>&1;\
	   if (${DIFF} output/ex2_1.out ex2_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with runex2, diffs above \n========================================="; fi ;\
	   ${RM} -f ex2_1.tmp

runex3:
	-@${MPIEXEC} -n 1 ./ex3 -nele 10 > ex3_1.tmp 2>&1;\
	   if (${DIFF} output/ex3_1.out ex3_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with runex4, diffs above \n========================================="; fi ;\
	   ${RM} -f ex3_1.tmp

runex4:
	-@${MPIEXEC} -n 1 ./ex4 -n 50 > ex4_1.tmp 2>&1;\
	   if (${DIFF} output/ex4_1.out ex4_1.tmp) then true ;  \
	   else echo ${PWD} ; echo "Possible problem with with runex4, diffs above \n========================================="; fi ;\
	   ${RM} -f ex4_1.tmp


TESTEXAMPLES_MOAB = ex1.PETSc runex1 ex1.rm ex2.PETSc runex2 ex2.rm

include ${PETSC_DIR}/conf/test
