
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tests/
EXAMPLESC       = ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c ex25.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_TS_LIB}
	${RM} ex1f.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_TS_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_TS_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_TS_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_TS_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_TS_LIB}
	${RM} ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o ${PETSC_TS_LIB}
	${RM} ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o ${PETSC_TS_LIB}
	${RM} ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o ${PETSC_TS_LIB}
	${RM} ex9.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o ${PETSC_TS_LIB}
	${RM} ex10.o

ex22: ex22.o  chkopts
	-${CLINKER} -o ex22 ex22.o  ${PETSC_TS_LIB}
	${RM} ex22.o

ex23: ex23.o  chkopts
	-${CLINKER} -o ex23 ex23.o  ${PETSC_TS_LIB}
	${RM} ex23.o

ex25: ex25.o  chkopts
	-${CLINKER} -o ex25 ex25.o ${PETSC_TS_LIB}
	${RM} ex25.o

#----------------------------------------------------------------------------------

runex5:
	-@${MPIEXEC} -n 1 ./ex5 -ts_max_steps 130 -monitor_interval 60 > ex5.tmp 2>&1;	  \
	   ${DIFF} output/ex5.out ex5.tmp || printf "${PWD}\nPossible problem with ex5_1, diffs above\n=========================================\n"; \
	   ${RM} -f ex5.tmp

runex5_2:
	-@${MPIEXEC} -n 4 ./ex5 -ts_max_steps 130 -monitor_interval 60 > ex5.tmp 2>&1;	  \
	   ${DIFF} output/ex5.out ex5.tmp || printf "${PWD}\nPossible problem with ex5_2, diffs above\n=========================================\n"; \
	   ${RM} -f ex5.tmp


TESTEXAMPLES_C_NOCOMPLEX_NOTSINGLE  = ex5.PETSc runex5 runex5_2 ex5.rm

include ${PETSC_DIR}/lib/petsc/conf/test
