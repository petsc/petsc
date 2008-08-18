# $Id$ 

TAO_DIR=/home/sarich/software/tao
PETSC_ARCH=linux-gnu-cxx-debug

ALL: eptorsion1

test_lmvmmat1: test_lmvmmat1.o tao_chkopts
	-${CLINKER} -o test_lmvmmat1 test_lmvmmat1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} test_lmvmmat1.o


runtest_lmvmmat1: 
	-@${MPIEXEC} -np 1 ./test_lmvmmat1 > test_lmvmmat1_1.tmp 2>&1; \
	if (${DIFF} output/test_lmvmmat1_1.out test_lmvmmat1_1.tmp) then true;\
        else echo "Possible problem with test_lmvmmat1_1, diffs above"; fi; \
	${RM} -f test_lmvmmat1_1.tmp

include ${TAO_DIR}/bmake/tao_common
