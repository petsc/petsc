LOCDIR=tests/
EXAMPLESC = linktest1.c linesearch1.c
TESTEXAMPLES_C = linktest1.tao runlinktest1 linktest1.rm clean \
         linesearchtest1.tao runlinesearchtest1 linesearchtest1.rm clean 


include ${TAO_DIR}/conf/tao_base

linktest1: linktest1.o tao_chkopts
	-${CLINKER} -o linktest1 linktest1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} linktest1.o


linesearchtest1: linesearchtest1.o tao_chkopts
	-${CLINKER} -o linesearchtest1 linesearchtest1.o ${TAO_LIB} ${PETSC_LIB}
	${RM} linesearchtest1.o


runlinktest1: 
	-@${MPIEXEC} -np 1 ./linktest1 > linktest1_1.tmp 2>&1; \
	if (${DIFF} output/linktest1_1.out linktest1_1.tmp) then true;\
        else echo "Possible problem with linktest1_1, diffs above"; fi; \
	${RM} -f linktest1_1.tmp

runlinesearchtest1: 
	-@${MPIEXEC} -np 1 ./linesearchtest1 > linesearchtest1_1.tmp 2>&1; \
	if (${DIFF} output/linesearchtest1_1.out linesearchtest1_1.tmp) then true;\
        else echo "Possible problem with linesearchtest1_1, diffs above"; fi; \
	${RM} -f linesearchtest1_1.tmp


