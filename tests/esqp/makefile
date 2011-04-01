LOCDIR=tests/
EXAMPLESC = test_link1.c linesearch1.c

include ${TAO_DIR}/conf/tao_base

loadq: loadq.o tao_chkopts
	-${CLINKER} -o loadq loadq.o ${TAO_LIB} ${PETSC_LIB}
	${RM} loadq.o

esqp: esqp.o tao_chkopts
	-${CLINKER} -o esqp esqp.o ${TAO_LIB} ${PETSC_LIB}
	${RM} esqp.o
