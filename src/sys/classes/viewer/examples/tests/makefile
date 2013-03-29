
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR  = src/sys/classes/viewer/examples/tests/
EXAMPLESC       = ex3.c ex4.c
MANSEC          = Viewer

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex3: ex3.o   chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_SYS_LIB}
	${RM} -f ex3.o

ex4: ex4.o   chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_SYS_LIB}
	${RM} -f ex4.o

runex3:
	-@${MPIEXEC} -n 1 ./ex3

runex4:
	-@${MPIEXEC} -n 1 ./ex4 -myviewer ascii:ex4w1.tmp ;\
	  if (${DIFF} output/ex4w.out ex4w1.tmp) then true; \
		else echo  ${PWD} ; echo "\nPossible problem with ex4, diffs above \n========================================="; fi;\
	  ${MPIEXEC} -n 1 ./ex4 -myviewer ascii:ex4w2.tmp:: ;\
	  if (${DIFF} output/ex4w.out ex4w2.tmp) then true; \
		else echo  ${PWD} ; echo "\nPossible problem with ex4, diffs above \n========================================="; fi;\
	  ${MPIEXEC} -n 1 ./ex4 -myviewer ascii:ex4w3.tmp::w ; \
	  if (${DIFF} output/ex4w.out ex4w3.tmp) then true; \
		else echo  ${PWD} ; echo "\nPossible problem with ex4, diffs above \n========================================="; fi;\
	  ${MPIEXEC} -n 1 ./ex4 -myviewer ascii:ex4w4.tmp::write ; \
	  if (${DIFF} output/ex4w.out ex4w4.tmp) then true; \
		else echo  ${PWD} ; echo "\nPossible problem with ex4, diffs above \n========================================="; fi;\
	  ${MPIEXEC} -n 1 ./ex4 -myviewer ascii:ex4a1.tmp::a ; \
	  if (${DIFF} output/ex4a.out ex4a1.tmp) then true; \
		else echo  ${PWD} ; echo "\nPossible problem with ex4, diffs above \n========================================="; fi;\
	  ${MPIEXEC} -n 1 ./ex4 -myviewer ascii:ex4a2.tmp::append ; \
	  if (${DIFF} output/ex4a.out ex4a2.tmp) then true; \
		else echo  ${PWD} ; echo "\nPossible problem with ex4, diffs above \n========================================="; fi;\
	  ${RM} -f ex4w1.tmp ex4w2.tmp ex4w3.tmp ex4w4.tmp ex4a1.tmp ex4a2.tmp

TESTEXAMPLES_C_X	  =
TESTEXAMPLES_C_X_MPIUNI = ex3.PETSc runex3 ex3.rm ex4.PETSc runex4 ex4.rm

include ${PETSC_DIR}/conf/test

