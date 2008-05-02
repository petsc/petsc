LOCDIR=tests/
EXAMPLESC = test_link1.c linesearch1.c
TESTEXAMPLES_C = test_link1.tao runtest_link1 test_link1.rm clean \
         test_linesearch1.tao runtest_linesearch1 test_linesearch1.rm \
         test_lmvmmat1.tao runtest_lmvmmat1 test_lmvmmat1.rm \
	 clean 


include ${TAO_DIR}/conf/tao_base

test_link1: test_link1.o tao_chkopts
	-${CLINKER} -o test_link1 test_link1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_link1.o


test_linesearch1: test_linesearch1.o tao_chkopts
	-${CLINKER} -o test_linesearch1 test_linesearch1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_linesearch1.o

test_lmvmmat1: test_lmvmmat1.o tao_chkopts
	-${CLINKER} -o test_lmvmmat1 test_lmvmmat1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_lmvmmat1.o



runtest_link1: 
	-@${MPIEXEC} -np 1 ./test_link1 > test_link1_1.tmp 2>&1; \
	if (${DIFF} output/test_link1_1.out test_link1_1.tmp) then true;\
        else echo "Possible problem with test_link1_1, diffs above"; fi; \
	${RM} -f test_link1_1.tmp

runtest_linesearch1: 
	-@${MPIEXEC} -np 1 ./test_linesearch1 > test_linesearch1_1.tmp 2>&1; \
	if (${DIFF} output/test_linesearch1_1.out test_linesearch1_1.tmp) then true;\
        else echo "Possible problem with test_linesearch1_1, diffs above"; fi; \
	${RM} -f test_linesearch1_1.tmp

runtest_lmvmmat1: 
	-@${MPIEXEC} -np 1 ./test_lmvmmat1 > test_lmvmmat1_1.tmp 2>&1; \
	if (${DIFF} output/test_lmvmmat1_1.out test_lmvmmat1_1.tmp) then true;\
        else echo "Possible problem with test_lmvmmat1_1, diffs above"; fi; \
	${RM} -f test_lmvmmat1_1.tmp


